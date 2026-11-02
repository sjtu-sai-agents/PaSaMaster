# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
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
class PaperNode:
    def __init__(self, attrs):
        self.title        = attrs.get("title", "")
        self.arxiv_id     = attrs.get("arxiv_id", "")
        self.depth        = attrs.get("depth", -1)
        self.child        = {k: [PaperNode(i) for i in v] for k, v in attrs.get("child", {}).items()}
        self.abstract     = attrs.get("abstract", "")
        self.description  = attrs.get("description", "")
        self.sections     = attrs.get("sections", "")      # section name -> list of citation papers
        self.source       = attrs.get("source", "Root")    # Root, Search, Expand
        self.select_score = attrs.get("select_score", -1) # the result of the selecte model
        self.extra        = attrs.get("extra", {})
        self.relevant_section = attrs.get("relevant_section", {})  # section name -> section content (title + content)
        self.relevant_section_ref = attrs.get("relevant_section_ref", {})  # section name -> list of expanded references
        self.reason = attrs.get("reason", "")  # section name -> section content (title + content)
        self.journal = attrs.get("journal", "")  
        self.year = attrs.get("year", "")  
        self.citations = attrs.get("citations", -1)  # list of papers that cite this paper
        self.authors = attrs.get("authors", ""),
        self.h5 = attrs.get("h5","")
        self.IF = attrs.get("IF","")
        self.ccf = attrs.get("CCF","")
        self.doi = attrs.get("doi","")
        self.origin_score = attrs.get("origin_score", -1)
        self.url = attrs.get("url", "")
        
    def todic(self):
        result = {
            "title":        self.title,
            "arxiv_id":     self.arxiv_id,
            "depth":        self.depth,
            "child":        {k: [i.todic() for i in v] for k, v in self.child.items()},
            "abstract":     self.abstract,
            "description":  self.description,
            "sections":     self.sections,
            "source":       self.source,
            "select_score": self.select_score,
            "extra":        self.extra,
            "relevant_section": self.relevant_section,
            "relevant_section_ref": self.relevant_section_ref,
            "reason": self.reason,
            "journal": self.journal,
            "year": self.year,
            "authors": self.authors,
            "citations": self.citations,
            "h5": self.h5,
            "IF": self.IF,
            "CCF": self.ccf,
            "doi": self.doi,
            "origin_score": self.origin_score,
            "url": self.url
        }
        
        # Add the content_sections field (if present)
        if hasattr(self, 'content_sections'):
            result["content_sections"] = self.content_sections
            
        # If the output field exists, include it in the result as well
        if hasattr(self, 'output'):
            result["output"] = self.output
        
        # Add dynamic scoring fields (if present)
        score_fields = ['small_score', 'big_score', 'big_chunk_score', 'small_model_chunk_score',
                       'small_reason', 'big_reason', 'big_chunk_reason', 'small_model_chunk_reason']
        for field in score_fields:
            if hasattr(self, field):
                result[field] = getattr(self, field)
            
        return result

    @staticmethod
    def sort_paper(item):
        return item.select_score