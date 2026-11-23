"""Exercise the interactive token retry without an account or network access."""
import os
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from capture_processing.tools import ensure_hf_login


class HubError(Exception):
    def __init__(self, status):
        self.response = SimpleNamespace(status_code=status)


class MissingToken(Exception):
    pass


class HuggingFaceLoginTests(unittest.TestCase):
    def setUp(self):
        hub = ModuleType("huggingface_hub")
        errors = ModuleType("huggingface_hub.errors")
        errors.HfHubHTTPError = HubError
        errors.LocalTokenNotFoundError = MissingToken
        self.whoami = hub.whoami = Mock()
        self.login = hub.login = Mock()
        for context in (
            patch.dict(sys.modules, {"huggingface_hub": hub, "huggingface_hub.errors": errors}),
            patch.dict(os.environ, {"HF_TOKEN": "", "HUGGING_FACE_HUB_TOKEN": ""}),
            patch.object(sys.stdin, "isatty", return_value=True),
        ):
            context.start()
            self.addCleanup(context.stop)

    def test_mistyped_token_reprompts_then_persists_only_the_accepted_token(self):
        self.whoami.side_effect = [MissingToken(), HubError(401), {"name": "tester"}]
        with patch.object(ensure_hf_login.getpass, "getpass", side_effect=["wrong", "valid"]) as prompt:
            ensure_hf_login.main()
        self.assertEqual(prompt.call_count, 2)
        self.login.assert_called_once_with(token="valid", add_to_git_credential=False)

    def test_server_failure_is_not_a_credential_retry(self):
        self.whoami.side_effect = [MissingToken(), HubError(503)]
        with patch.object(ensure_hf_login.getpass, "getpass", return_value="token") as prompt:
            with self.assertRaises(HubError) as error:
                ensure_hf_login.main()
        self.assertEqual(error.exception.response.status_code, 503)
        self.assertEqual(prompt.call_count, 1)
        self.login.assert_not_called()
