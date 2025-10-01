__all__ = ["Reports"]

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from ..utils.namespaces import LazyNamespace
from ..utils.report_utils import (
    copy_report_resources,
    generate_zipped_report,
    run_quarto,
)
from .ExplanationsUtils import _DEFAULT

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .Explanations import Explanations

ENCODING = "utf-8"
UNIQUE_CONTEXTS_FILENAME = "unique_contexts.json"
PLOTS_FOR_BATCH = "plots_for_batch"
CONTEXT_FOLDER = "by-context"

# init template folder and filenames
# these are the templates used to generate the context files
# and the overview file
TEMPLATES_FOLDER = "assets/templates"
INTRODUCTION_FILENAME = "getting-started.qmd"
OVERVIEW_FILENAME = "overview.qmd"
ALL_CONTEXT_HEADER_TEMPLATE = "all_context_header.qmd"
ALL_CONTEXT_CONTENT_TEMPLATE = "all_context_content.qmd"
SINGLE_CONTEXT_TEMPLATE = "context.qmd"

class Reports(LazyNamespace):
    dependencies = ["yaml"]
    dependency_group = "explanations"

    def __init__(self, explanations: "Explanations"):
        self.explanations = explanations

        self.report_foldername = "reports"
        self.report_folderpath = os.path.join(
            self.explanations.root_dir, self.report_foldername
        )
        self.report_output_dir = os.path.join(self.report_folderpath, "_site")

        self.aggregate_folder = self.explanations.aggregate.data_folderpath
        self.params_file = os.path.join(self.report_folderpath, "scripts", "params.yml")

        self.contexts = None
        super().__init__()

    def generate(
        self,
        report_filename: str = "explanations_report.zip",
        top_n: int = _DEFAULT.TOP_N.value,
        top_k: int = _DEFAULT.TOP_K.value,
        zip_output: bool = False,
        verbose: bool = False,
    ):
        """Generate the explanations report.

        Args:
            report_filename (str):
                Name of the output report file.
            top_n (int):
                Number of top explanations to include.
            top_k (int):
                Number of top features to include in explanations.
            zip_output (bool):
                Whether to zip the output report.
                The filename will be used as the zip file name.
            verbose (bool):
                Whether to print verbose output during report generation.
        """
        try:
            self.explanations.aggregate.validate_folder()
        except Exception as e:
            logger.error("Validation failed: %s", e)
            raise

        self._validate_report_dir()

        try:
            self._copy_report_resources()
        except (OSError, shutil.Error) as e:
            logger.error("IO error during resource copy: %s", e)
            raise

        self._set_params(top_n=top_n, top_k=top_k, verbose=verbose)

        self._generate_batchdirs() 
        # if return_code != 0:
        #     logger.error("Quarto command failed with return code %s", return_code)
        #     raise RuntimeError(f"Quarto command failed with return code {return_code}")

        # if zip_output:
        #     generate_zipped_report(report_filename, self.report_output_dir)

    def _generate_batchdirs(
            self,
            top_n: int = _DEFAULT.TOP_N.value,
            top_k: int = _DEFAULT.TOP_K.value):
        report_generator = self.ReportGenerator(
            self.report_folderpath,
            self.aggregate_folder.name,
            top_n=top_n,
            top_k=top_k)
        report_generator.run()

        return report_generator.contexts

    def _validate_report_dir(self):
        if not os.path.exists(self.report_folderpath):
            os.makedirs(self.report_folderpath, exist_ok=True)

    def _copy_report_resources(self):
        copy_report_resources(
            resource_dict=[
                ("GlobalExplanations", self.report_folderpath),
                ("assets", os.path.join(self.report_folderpath, "assets")),
            ],
        )

    def _set_params(
        self,
        top_n: int = _DEFAULT.TOP_N.value,
        top_k: int = _DEFAULT.TOP_K.value,
        verbose: bool = False,
    ):
        params = {}
        params["top_n"] = top_n
        params["top_k"] = top_k
        params["verbose"] = verbose
        params["data_folder"] = self.aggregate_folder.name

        with open(self.params_file, "w", encoding="utf-8") as file:
            yaml.safe_dump(params, file)

    class ReportGenerator:

        def __init__(
                self,
                report_folder: str,
                data_folder: str,
                top_n: int = _DEFAULT.TOP_N.value,
                top_k: int = _DEFAULT.TOP_K.value):

            self.report_folder = report_folder
            self.top_n = top_n
            self.top_k = top_k

            self.by_context_folder = f"{self.report_folder}/{CONTEXT_FOLDER}"
            if not os.path.exists(self.by_context_folder):
                os.makedirs(self.by_context_folder, exist_ok=True)

            self.contexts = None

            self.root_dir = os.path.abspath(os.path.join(self.report_folder, ".."))
            self.data_folder = os.path.abspath(
                os.path.join(self.report_folder, "..", data_folder)
            )

        @staticmethod
        def _get_context_dict(context_info: str) -> dict:
            return json.loads(context_info)["partition"]

        def _get_context_string(self, context_info: str) -> str:
            return "-".join(
                [
                    v.replace(" ", "")
                    for _, v in self._get_context_dict(context_info).items()
                ]
            )

        def _read_template(self, template_filename: str) -> str:
            """Read a template file and return its content."""
            with open(
                f"{self.report_folder}/{TEMPLATES_FOLDER}/{template_filename}", "r", encoding=ENCODING
            ) as fr:
                return fr.read()

        def _write_single_context_file(
            self,
            embed_path_for_batch: str,
            filename: str,
            template: str,
            context_str: str,
            context_label: str,
        ):
            with open(filename, "w", encoding=ENCODING) as fw:
                f_context_template = f"""{
                    template.format(
                        EMBED_PATH_FOR_BATCH=embed_path_for_batch,
                        CONTEXT_STR=context_str,
                        CONTEXT_LABEL=context_label,
                        TOP_N=self.top_n,
                    )
                }"""
                fw.write(f_context_template)

        def _write_single_main_context_file(
                self,
                embed_file: str,
                filename: str,
                template: str,
                context_str: str
        ):
            with open(filename, "w", encoding=ENCODING) as fw:
                f_context_template = f"""{
                    template.format(
                        EMBED_FILE=embed_file,
                        CONTEXT_STR=context_str,
                    )
                }"""
                fw.write(f_context_template)

        def _write_header_to_file(self, file_batch_nb: str, filename: str):
            template = self._read_template(ALL_CONTEXT_HEADER_TEMPLATE)

            f_template = f"""{
                template.format(
                    ROOT_DIR=self.root_dir,
                    DATA_FOLDER=self.data_folder,
                    DATA_PATTERN=f"*_BATCH_{file_batch_nb}.parquet",
                    TOP_N=self.top_n,
                )
            }"""

            with open(filename, "w", encoding=ENCODING) as writer:
                writer.write(f_template)

        def _append_content_to_file(
            self,
            filename: str,
            template: str,
            context_dict: dict,
            context_label: str,
        ):
            with open(filename, "a", encoding=ENCODING) as writer:
                f_content_template = f"""{
                    template.format(
                        CONTEXT_DICT=context_dict,
                        CONTEXT_LABEL=context_label,
                        TOP_N=self.top_n,
                        TOP_K=self.top_k,
                    )
                }"""

                writer.write("\n")
                writer.write(f_content_template)

        def _get_unique_contexts(self):
            if self.contexts is not None:
                return self.contexts

            unique_contexts_file = f"{self.data_folder}/{UNIQUE_CONTEXTS_FILENAME}"
            if not os.path.exists(unique_contexts_file):
                raise FileNotFoundError(
                    f"Unique contexts file not found: {unique_contexts_file}. "
                    "Please ensure that aggregates have been generated."
                )
            with open(unique_contexts_file, "r", encoding=ENCODING) as f:
                self.contexts = json.load(f)
            return self.contexts

        def _generate_for_selected_contexts_qmds(
                self,
                context_content_template: str,
                single_context_template: str,
                single_main_context_template: str,
                contexts: list[str],
                batch_folder: str,
                batch_embed_filename: str,
                batch_embed_filepath: str,
                ):
            something = {}
            for context in contexts:

                context_str = self._get_context_string(context)
                context_label = ("plt-" + context_str).lower()

                self._append_content_to_file(
                    filename=batch_embed_filepath,
                    template=context_content_template,
                    context_dict=self._get_context_dict(context),
                    context_label=context_label,
                )

                context_filename = f'{context_label}.qmd'
                context_filepath = f"{batch_folder}/{context_filename}"

                self._write_single_context_file(
                    embed_path_for_batch=batch_embed_filename,
                    filename=context_filepath,
                    template=single_context_template,
                    context_str=context_str,
                    context_label=context_label,
                )

                self._write_single_main_context_file(
                    embed_file=f'_site/{context_label}.html',
                    filename=f"{self.by_context_folder}/{context_filename}",
                    template=single_main_context_template,
                    context_str=context_str
                )

                something[context_str] = f"{CONTEXT_FOLDER}/{context_label}.html"
            return something

        def _move_files(self):
            src_site_folder = os.path.join(self.by_context_folder, "_site")
            dst_site_folder = os.path.join(self.report_folder, "_site", CONTEXT_FOLDER)
            os.makedirs(dst_site_folder, exist_ok=True)
            for filename in os.listdir(src_site_folder):
                if filename.startswith("plt-") and filename.endswith(".html"):
                    shutil.move(
                        os.path.join(src_site_folder, filename),
                        os.path.join(dst_site_folder, filename)
                    )

        def _generate_by_context_qmds(self):
            contexts = self._get_unique_contexts()

            something = []
            for file_batch_nb, context_batches in contexts.items():
                batch_folder = f"{self.by_context_folder}/{PLOTS_FOR_BATCH}_{file_batch_nb}"
                os.makedirs(batch_folder, exist_ok=True)

                # copy the _quarto.yml file to the batch folder
                shutil.copyfile(
                    f"{self.report_folder}/{TEMPLATES_FOLDER}/_quarto.yml",
                    f"{batch_folder}/_quarto.yml",
                )

                batch_embed_filename = f"{PLOTS_FOR_BATCH}_{file_batch_nb}.qmd"
                batch_embed_filepath = f"{batch_folder}/{batch_embed_filename}"

                # write header
                self._write_header_to_file(file_batch_nb, batch_embed_filepath)

                # write content
                context_content_template = self._read_template(ALL_CONTEXT_CONTENT_TEMPLATE)
                single_context_template = self._read_template(SINGLE_CONTEXT_TEMPLATE)
                single_main_context_template = self._read_template("main_site_context.qmd")

                for query_batch_nb, selected_contexts in context_batches.items():
                    something.append(self._generate_for_selected_contexts_qmds(
                        context_content_template=context_content_template,
                        single_context_template=single_context_template,
                        single_main_context_template=single_main_context_template,
                        contexts=selected_contexts,
                        batch_folder=batch_folder,
                        batch_embed_filename=batch_embed_filename,
                        batch_embed_filepath=batch_embed_filepath
                    ))

                # run quarto to generate the site for this batch
                try:
                    return_code = run_quarto(
                        temp_dir=Path(batch_folder),
                        output_type=None)
                except subprocess.CalledProcessError as e:
                    logger.error("Quarto command failed: %s", e)
                    raise

            # read the main yml file
            self._update_main_yml_file(something)

            # move the generated by-context/_site folder under reports/_site/by-context
            # self._move_files()

        def _update_main_yml_file(self, something: list[dict]):
            with open(f"{self.report_folder}/_quarto.yml", "r", encoding=ENCODING) as fr:
                main_yml = yaml.safe_load(fr)
            
            doitlady = []
            for ll in something:
                for k, v in ll.items():
                    doitlady.append({"text": k, "href": v})
            section = {"section": "By Context", "contents": doitlady}

            main_yml['website']['sidebar']['contents'].append(section)

            # write back the main yml file
            with open(f"{self.report_folder}/_quarto.yml", "w", encoding=ENCODING) as fw:
                yaml.safe_dump(main_yml, fw)

        def _generate_overview_qmd(self):
            with open(
                f"{self.report_folder}/{TEMPLATES_FOLDER}/{OVERVIEW_FILENAME}", "r", encoding=ENCODING
            ) as fr:
                template = fr.read()

            f_template = f"""{
                template.format(
                    ROOT_DIR=self.root_dir,
                    DATA_FOLDER=self.data_folder,
                    TOP_N=self.top_n,
                    TOP_K=self.top_k,
                )
            }
            """

            with open(f"{self.report_folder}/{OVERVIEW_FILENAME}", "w", encoding=ENCODING) as f:
                f.write(f_template)

        def _generate_introduction_qmd(self):
            with open(
                f"{self.report_folder}/{TEMPLATES_FOLDER}/{INTRODUCTION_FILENAME}", "r", encoding=ENCODING
            ) as fr:
                template = fr.read()

            f_template = f"""{
                template.format(
                    TOP_N=self.top_n,
                    TOP_K=self.top_k,
                )
            }"""

            with open(f"{self.report_folder}/{INTRODUCTION_FILENAME}", "w", encoding=ENCODING) as f:
                f.write(f_template)

        def run(self):
            """Main method to generate the report files."""
            self._generate_introduction_qmd()
            logger.info("Generated introduction QMD file.")

            self._generate_overview_qmd()
            logger.info("Generated overview QMD file.")

            self._generate_by_context_qmds()
            logger.info("Generated by-context QMDs files.")

            try:
                return_code = run_quarto(
                    temp_dir=Path(self.report_folder),
                    output_type=None)
            except subprocess.CalledProcessError as e:
                logger.error("Quarto command failed: %s", e)
                raise
