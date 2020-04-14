import pytest

import A_3_1_FaceRecognition_Various, A_3_2_FaceRecognition_Various

# To run, open cmd in folder and type:
    # All tests: pytest -vv --durations=0
    # Specific test: pytest -vv --durations=0 -k "method name"
    # Ref: https://docs.pytest.org/en/latest/usage.html


def test_2_0_FaceRecognition_Various():
    execfile('A_3_1_FaceRecognition_Various')
    assert True

def A_3_2_FaceRecognition_Various():
    execfile('A_3_2_FaceRecognition_Various')
    assert True

