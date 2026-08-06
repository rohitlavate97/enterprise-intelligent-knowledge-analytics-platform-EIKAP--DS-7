"""Restricted module for HR resume screening and job description matching."""

import re
from typing import Dict, Any, List, Optional
from datetime import datetime

from shared.logging import get_logger

logger = get_logger(__name__)

class ResumeScreener:
    """RESTRICTED maturity module for HR resume screening."""

    def __init__(self):
        """Initialize the ResumeScreener."""
        logger.info("Initializing ResumeScreener (Restricted)")
        self.gender_pronouns = {"he", "him", "his", "she", "her", "hers"}
        self.location_indicators = {"zip", "address", "street", "city", "state", "country"}
    
    def screen_resume(self, resume_text: str, job_description: str, required_skills: Optional[List[str]] = None) -> Dict[str, Any]:
        """Screen resume against job description and required skills. Framing for human recruiter review."""
        logger.info("Screening resume")
        
        resume_lower = resume_text.lower()
        job_lower = job_description.lower()
        
        # 1. Fairness Audit
        audit = self._perform_fairness_audit(resume_lower)
        
        # 2. Skill Matching
        if required_skills is None:
            required_skills = self._extract_skills(job_lower)
            
        matched_skills = [skill for skill in required_skills if skill.lower() in resume_lower]
        missing_skills = [skill for skill in required_skills if skill.lower() not in resume_lower]
        
        match_score = 0.0
        if required_skills:
            match_score = (len(matched_skills) / len(required_skills)) * 100.0
        else:
            match_score = 75.0
            
        # 3. Recommendation Framing (NO automated decision)
        if match_score >= 70.0 and audit["audit_pass"]:
            recommendation = "shortlist_for_recruiter_review"
        else:
            recommendation = "manual_recruiter_review"
            
        return {
            "match_score": match_score,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "fairness_audit": audit,
            "recommendation": recommendation,
            "summary": f"Resume match score is {match_score:.1f}%. Recommended for {recommendation}."
        }

    def match(self, resume_text: str, job_description: str) -> float:
        """Alias for screen_resume returning normalized match score (0.0 to 1.0)."""
        res = self.screen_resume(resume_text, job_description)
        return float(res["match_score"] / 100.0)

    def audit_fairness(self, text: str) -> Dict[str, Any]:
        """Alias returning fairness audit."""
        return self._perform_fairness_audit(text.lower())

    def review_framing(self, text: str) -> Dict[str, Any]:
        """Alias returning human recruiter review framing."""
        res = self.screen_resume(text, "")
        return {
            "recommendation": res["recommendation"],
            "summary": res["summary"]
        }

    def _perform_fairness_audit(self, resume_lower: str) -> Dict[str, Any]:
        """Check for potential bias indicators."""
        words = set(re.findall(r'\b\w+\b', resume_lower))
        
        # Gender bias check
        gender_bias_flag = any(pronoun in words for pronoun in self.gender_pronouns)
        
        # Location bias check
        location_bias_flag = any(loc in words for loc in self.location_indicators)
        
        # Age proxy check (graduations > 20 years ago or older dates)
        age_proxy_flag = False
        current_year = datetime.now().year
        years_found = re.findall(r'\b(19\d{2}|20\d{2})\b', resume_lower)
        for year_str in years_found:
            year = int(year_str)
            if current_year - year > 20:
                age_proxy_flag = True
                break
                
        audit_pass = not (gender_bias_flag or location_bias_flag or age_proxy_flag)
        
        return {
            "gender_bias_flag": gender_bias_flag,
            "age_proxy_flag": age_proxy_flag,
            "location_bias_flag": location_bias_flag,
            "audit_pass": audit_pass
        }

    def _extract_skills(self, text: str) -> List[str]:
        """Simple heuristic to extract potential skills."""
        common_skills = ["python", "java", "c++", "sql", "machine learning", "nlp", "communication", "leadership", "aws", "docker"]
        return [skill for skill in common_skills if skill in text]
