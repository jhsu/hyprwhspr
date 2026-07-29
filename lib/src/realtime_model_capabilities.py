"""Capabilities for OpenAI realtime transcription model variants.

Keep model-specific protocol details in one place so the transport, backend
validation, and partial-preview UI do not drift apart.
"""

from typing import Dict


_DEFAULT_CAPABILITIES = {
    'language_field': 'language',
    'supports_delay': False,
    'manual_commit': False,
    'supports_prompt': False,
    'supports_keywords': False,
    'supports_partial_preview': False,
    'transcription_only': False,
}


REALTIME_TRANSCRIPTION_CAPABILITIES = {
    # Legacy OpenAI realtime transcription model retained for compatibility.
    'gpt-realtime-whisper': {
        'supports_delay': True,
        'manual_commit': True,
        'supports_partial_preview': True,
        'transcription_only': True,
    },
    # Current OpenAI realtime transcription model.
    'gpt-live-transcribe': {
        'language_field': 'languages',
        'supports_delay': True,
        'manual_commit': True,
        'supports_prompt': True,
        'supports_keywords': True,
        'supports_partial_preview': True,
        'transcription_only': True,
    },
}


def get_realtime_model_capabilities(model_id: str) -> Dict:
    """Return protocol capabilities for a realtime model identifier."""
    capabilities = _DEFAULT_CAPABILITIES.copy()
    capabilities.update(REALTIME_TRANSCRIPTION_CAPABILITIES.get(model_id, {}))
    return capabilities