"""
PDF Resume/CV Parser Module

Extracts text from PDF files and uses LLM to convert unstructured
resume content into structured employee profiles.
"""

import json
import uuid
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ParsedProfile:
    """Structured profile extracted from a resume/CV."""
    id: str
    name: str
    role: str
    constraints: Dict[str, Any]
    metadata: Dict[str, Any]
    technical: Dict[str, Any]
    personality: Dict[str, Any]
    collaboration: Dict[str, Any]
    learning_behavior: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class PDFParser:
    """
    Parses PDF resumes/CVs and extracts structured profiles.
    
    Uses a hybrid approach:
    1. PyMuPDF (fitz) extracts raw text from PDF
    2. LLM structures the text into our profile schema
    
    Example:
        parser = PDFParser(llm_client)
        profile = parser.parse_resume(pdf_bytes)
    """
    
    # Profile schema for LLM extraction
    EXTRACTION_PROMPT = '''
You are an expert HR data extraction system. Extract structured information from this resume/CV text.

RESUME TEXT:
"""
{resume_text}
"""

Extract and return a JSON object with this EXACT structure:
{{
    "name": "Full name of the candidate",
    "role": "Primary job title/role (e.g., 'Developer, full-stack', 'Data scientist')",
    "metadata": {{
        "dev_type": "Developer type matching: Developer, full-stack | Developer, back-end | Developer, front-end | Developer, mobile | DevOps specialist | Data scientist or machine learning specialist | Cloud infrastructure engineer | System administrator",
        "work_experience_years": "Total years of experience as a number string (e.g., '5.0')",
        "years_code": "Years coding as a number string",
        "employment": "Employment type: Employed, full-time | Independent contractor | Employed, part-time",
        "org_size": "Last known organization size or 'Unknown'",
        "industry": "Primary industry: Healthcare | Fintech | E-commerce | Education | Consulting | Cybersecurity | Manufacturing | Media | Telecommunications | General"
    }},
    "technical": {{
        "skills": ["List", "of", "programming", "languages", "and", "frameworks"],
        "tools": ["List", "of", "tools", "platforms", "and", "software"]
    }},
    "personality": {{
        "Belbin_team_role": "Infer from resume: Plant | Resource Investigator | Co-ordinator | Shaper | Monitor Evaluator | Teamworker | Implementer | Completer Finisher | Specialist"
    }},
    "collaboration": {{
        "communication_style": "Infer: Direct | Indirect | Mixed | Formal | Informal",
        "conflict_style": "Default to 'Collaborate'",
        "leadership_preference": "Infer: Lead | Follow | Co-lead | Autonomous",
        "deadline_discipline": "Default to 'Flexible'"
    }},
    "learning_behavior": {{
        "learning_orientation": "Infer: Low | Medium | High | Very High",
        "knowledge_sharing": "Default to 'Medium'"
    }}
}}

IMPORTANT:
- Extract ALL technical skills mentioned (programming languages, frameworks, libraries)
- Extract ALL tools mentioned (IDEs, cloud platforms, databases, DevOps tools)
- If information is not available, use reasonable defaults
- Return ONLY valid JSON, no markdown formatting
'''

    def __init__(self, llm_client, llm_model: str):
        """
        Initialize PDF parser with LLM client.
        
        Args:
            llm_client: OpenAI-compatible client
            llm_model: Model identifier (e.g., 'gpt-4o-mini')
        """
        self.llm_client = llm_client
        self.llm_model = llm_model
        self._fitz = None  # Lazy load PyMuPDF
    
    def _get_fitz(self):
        """Lazy load PyMuPDF to avoid import errors if not installed."""
        if self._fitz is None:
            try:
                import fitz
                self._fitz = fitz
            except ImportError:
                raise ImportError(
                    "PyMuPDF (fitz) is required for PDF parsing. "
                    "Install with: pip install PyMuPDF"
                )
        return self._fitz
    
    def extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        """
        Extract raw text from PDF bytes.
        
        Args:
            pdf_bytes: PDF file content as bytes
            
        Returns:
            Extracted text content
        """
        fitz = self._get_fitz()
        
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text_parts = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text_parts.append(page.get_text())
            
            doc.close()
            return "\n".join(text_parts)
            
        except Exception as e:
            logger.error(f"PDF text extraction failed: {e}")
            raise ValueError(f"Could not extract text from PDF: {e}")
    
    def extract_text_from_file(self, file_path: str) -> str:
        """
        Extract text from a PDF file path.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        with open(file_path, "rb") as f:
            return self.extract_text_from_pdf(f.read())
    
    def _call_llm(self, prompt: str, extra_headers: Dict = None) -> str:
        """Make LLM API call with error handling."""
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000,
                extra_headers=extra_headers or {}
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse and validate LLM JSON response."""
        # Clean up common LLM response issues
        cleaned = response.strip()
        cleaned = cleaned.replace("```json", "").replace("```", "")
        cleaned = cleaned.strip()
        
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Response was: {response[:500]}")
            raise ValueError("LLM returned invalid JSON")
    
    def structure_resume_text(
        self,
        resume_text: str,
        extra_headers: Dict = None
    ) -> ParsedProfile:
        """
        Use LLM to convert resume text into structured profile.
        
        Args:
            resume_text: Raw text extracted from resume
            extra_headers: Optional headers for API call
            
        Returns:
            ParsedProfile object
        """
        # Truncate very long resumes to avoid token limits
        max_chars = 8000
        if len(resume_text) > max_chars:
            resume_text = resume_text[:max_chars] + "\n[TRUNCATED]"
        
        prompt = self.EXTRACTION_PROMPT.format(resume_text=resume_text)
        response = self._call_llm(prompt, extra_headers)
        data = self._parse_llm_response(response)
        
        # Create profile with generated ID
        return ParsedProfile(
            id=str(uuid.uuid4()),
            name=data.get("name", "Unknown Candidate"),
            role=data.get("role", "Developer, full-stack"),
            constraints=data.get("constraints", {"weekly_availability_hours": "40"}),
            metadata=data.get("metadata", {}),
            technical=data.get("technical", {"skills": [], "tools": []}),
            personality=data.get("personality", {"Belbin_team_role": "Teamworker"}),
            collaboration=data.get("collaboration", {}),
            learning_behavior=data.get("learning_behavior", {})
        )
    
    def parse_resume(
        self,
        pdf_bytes: bytes,
        extra_headers: Dict = None
    ) -> ParsedProfile:
        """
        Full pipeline: Extract text from PDF and structure it.
        
        Args:
            pdf_bytes: PDF file content as bytes
            extra_headers: Optional headers for API call
            
        Returns:
            ParsedProfile object
        """
        logger.info("Extracting text from PDF...")
        text = self.extract_text_from_pdf(pdf_bytes)
        
        logger.info("Structuring resume with LLM...")
        profile = self.structure_resume_text(text, extra_headers)
        
        logger.info(f"Successfully parsed resume for: {profile.name}")
        return profile
    
    def parse_multiple_resumes(
        self,
        pdf_files: List[bytes],
        extra_headers: Dict = None
    ) -> List[ParsedProfile]:
        """
        Parse multiple PDF resumes.
        
        Args:
            pdf_files: List of PDF file contents as bytes
            extra_headers: Optional headers for API calls
            
        Returns:
            List of ParsedProfile objects
        """
        profiles = []
        
        for i, pdf_bytes in enumerate(pdf_files):
            try:
                logger.info(f"Processing resume {i+1}/{len(pdf_files)}...")
                profile = self.parse_resume(pdf_bytes, extra_headers)
                profiles.append(profile)
            except Exception as e:
                logger.error(f"Failed to parse resume {i+1}: {e}")
                continue
        
        return profiles
