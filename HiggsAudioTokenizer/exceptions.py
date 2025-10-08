# Copyright 2025 HiggsAudiov2-Unofficial Contributors
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

"""Custom exceptions for HiggsAudioTokenizer.

Provides specific exception types for different error scenarios to ensure
fail-fast behavior with clear error messages.
"""


class HiggsAudioError(Exception):
    """Base exception for all HiggsAudioTokenizer errors."""
    pass


class AudioLoadError(HiggsAudioError):
    """Raised when audio loading fails."""
    pass


class ConfigError(HiggsAudioError):
    """Raised for configuration-related errors."""
    pass


class CheckpointError(HiggsAudioError):
    """Raised when checkpoint loading or saving fails."""
    pass


class ModelError(HiggsAudioError):
    """Raised for model initialization or loading failures."""
    pass


class DatasetError(HiggsAudioError):
    """Raised for dataset-related errors."""
    pass


class ProcessingError(HiggsAudioError):
    """Raised for general processing failures."""
    pass


class ImportError(HiggsAudioError):
    """Raised when required dependencies are missing."""
    pass


class ValidationError(HiggsAudioError):
    """Raised when input validation fails."""
    pass