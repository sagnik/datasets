# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""The Stanford Natural Language Inference (SNLI) Corpus."""

from __future__ import absolute_import, division, print_function

import json
import os

import datasets


_CITATION = """\
something
"""

_DESCRIPTION = """\
something
"""

_DATA_URL = "https://github.com/text-machine-lab/quail/archive/master.zip"


class QaClassifyConfig(datasets.BuilderConfig):
    """BuilderConfig for QuAIL."""

    def __init__(self, **kwargs):
        """BuilderConfig for QuAIL.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(QaClassifyConfig, self).__init__(**kwargs)


class QaClassify(datasets.GeneratorBasedBuilder):
    """The Stanford Natural Language Inference (SNLI) Corpus."""

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="quail",
            version=datasets.Version("1.0.0", ""),
            description="QA SNLI jsonl",
        )
    ]

    def _info(self):
        if self.config.name == "quail":
            return datasets.DatasetInfo(
                description=_DESCRIPTION,
                features=datasets.Features(
                    {
                        "question": datasets.Value("string"),
                        "context": datasets.Value("string"),
                        "question_type": datasets.features.ClassLabel(names=['Entity_properties',
                                                                              'Character_identity',
                                                                              'Temporal_order',
                                                                              'Event_duration',
                                                                              'Causality',
                                                                              'Belief_states',
                                                                              'Subsequent_state',
                                                                              'Unanswerable',
                                                                              'Factual']
                                                                             ),
                    }
                ),
                # No default supervised_keys (as we have to pass both premise
                # and hypothesis as input).
                supervised_keys=None,
                homepage="https://github.com/text-machine-lab/quail",
                citation=_CITATION,
            )
        raise NotImplementedError()

    def _split_generators(self, dl_manager):
        if self.config.name == "quail":
            dl_dir = dl_manager.download_and_extract(_DATA_URL)
            data_dir = os.path.join(dl_dir, "quail-master/quail_v1.3/json/")
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TEST, gen_kwargs={"filepath": os.path.join(data_dir, "challenge.jsonl")}
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION, gen_kwargs={"filepath": os.path.join(data_dir, "dev.jsonl")}
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN, gen_kwargs={"filepath": os.path.join(data_dir, "train.jsonl")}
                ),
            ]
        raise NotImplementedError()

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        if self.config.name == "quail":
            with open(filepath, encoding="utf-8") as rf:
                idx = 0
                for line in rf:
                    idx += 1
                    datum = json.loads(line)
                    yield idx, {
                        "question": datum["question"],
                        "context": datum["context"],
                        "question_type": datum["question_type"]
                    }
        else:
            raise NotImplementedError()