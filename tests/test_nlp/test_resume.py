def test_resume_screener_matching(resume_screener, sample_resume, sample_job_description):
    score = resume_screener.match(sample_resume, sample_job_description)
    assert 0.0 <= score <= 1.0
    assert score > 0.0

def test_resume_screener_fairness_audit(resume_screener, sample_resume):
    audit = resume_screener.audit_fairness(sample_resume)
    assert "gender_bias_flag" in audit
    assert "age_proxy_flag" in audit
    assert audit["gender_bias_flag"] is True
    assert audit["age_proxy_flag"] is True

def test_resume_screener_human_review_framing(resume_screener, sample_resume):
    framing = resume_screener.review_framing(sample_resume)
    assert "action" not in framing
    assert "rejection" not in framing
    assert "summary" in framing
