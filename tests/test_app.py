
from app import analyze_resume


def test_matching_skills():
    resume = """
    I developed web applications using Python, Flask,
    SQL, and Git. I understand REST API development.
    """
    job = """
    Python developer required with Flask, SQL,
    Git, and REST API knowledge.
    """

    score, matched, missing = analyze_resume(resume, job)

    assert score > 0
    assert "python" in matched
    assert "flask" in matched
    assert "sql" in matched
    assert "git" in matched
    assert "rest api" in matched
    assert missing == []


def test_missing_skills():
    resume = "I know Python and HTML."
    job = "We need Python, Flask, and Docker."

    score, matched, missing = analyze_resume(resume, job)

    assert "python" in matched
    assert "flask" in missing
    assert "docker" in missing
