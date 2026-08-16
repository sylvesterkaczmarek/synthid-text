# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tests for Bayesian detector training device validation."""

from unittest import mock

from absl.testing import absltest
from synthid_text import detector_bayesian
import torch


class BayesianDetectorDeviceTest(absltest.TestCase):

  def testTrainBestDetectorRejectsCpu(self):
    with self.assertRaisesRegex(ValueError, 'Use GPU or TPU for training'):
      detector_bayesian.BayesianDetector.train_best_detector(
          tokenized_wm_outputs=mock.sentinel.watermarked_outputs,
          tokenized_uwm_outputs=mock.sentinel.unwatermarked_outputs,
          logits_processor=mock.sentinel.logits_processor,
          tokenizer=mock.sentinel.tokenizer,
          torch_device=torch.device('cpu'),
      )

  def testTrainBestDetectorAcceptsCudaDevice(self):
    processed = (
        mock.sentinel.train_g_values,
        mock.sentinel.train_masks,
        mock.sentinel.train_labels,
        mock.sentinel.cv_g_values,
        mock.sentinel.cv_masks,
        mock.sentinel.cv_labels,
    )
    expected = (mock.sentinel.detector, 0.25)

    with mock.patch.object(
        detector_bayesian.BayesianDetector,
        'process_raw_model_outputs',
        return_value=processed,
    ) as process_outputs, mock.patch.object(
        detector_bayesian.BayesianDetector,
        'train_best_detector_given_g_values',
        return_value=expected,
    ) as train_detector:
      result = detector_bayesian.BayesianDetector.train_best_detector(
          tokenized_wm_outputs=mock.sentinel.watermarked_outputs,
          tokenized_uwm_outputs=mock.sentinel.unwatermarked_outputs,
          logits_processor=mock.sentinel.logits_processor,
          tokenizer=mock.sentinel.tokenizer,
          torch_device=torch.device('cuda'),
      )

    self.assertEqual(result, expected)
    process_outputs.assert_called_once()
    train_detector.assert_called_once()


if __name__ == '__main__':
  absltest.main()
