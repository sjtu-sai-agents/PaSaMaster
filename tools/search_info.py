from config_loader import get_config_path
import json
import time
from typing import List, Optional, Dict, Any
import os
current_path = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config/config.json")
with open(get_config_path(), "r", encoding="utf-8") as f:
    data_path = json.load(f).get("meta_info_path", "")



class MetaInfoClient:
    def __init__(self):
        self.data_path = data_path
        self.dataset = self.load_dataset()

    def load_dataset(self) -> Dict[str, Dict[str, Any]]:
        """Load the entire dataset into memory and build a lowercase title -> record mapping."""
        title_to_record = {}
        if not self.data_path:
            return title_to_record
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line. strip()
                if not line:
                    continue
                record = json.loads(line)
                title = record.get("title")
                title_lower = title.lower()
                title_to_record[title_lower] = record
        return title_to_record

    def clear_title(self, title: str) -> str:
        """Strip leading/trailing spaces and newline characters from the title."""
        title = title.replace("\n ", "")
        return title.strip()

    def get_paper_by_title(self, titles: List[str]) -> List[Dict[str, Any]]:
        """Look up paper records by a list of titles (case-insensitive).
        
        For each title:
          - If found, return the original record + {"in_dataset": True}
          - If not found, return {"in_dataset": False}
        """
        results = []
        for title in titles:
            key = self.clear_title(title).lower()
            record = self.dataset.get(key)
            if record is not None:
                result_record = dict(record)
                result_record["title"] = title
                results.append(result_record)
            else:
                results.append(None)
        return results
    
    
client = MetaInfoClient()

def search_meta_info(titles):
    begin = time.time()
    records = client.get_paper_by_title(titles)
    end = time.time()
    return records
    

if __name__ == "__main__":
    titles = ["When Less is More: Investigating Data Pruning for Pretraining LLMs at\n  Scale"]
    result = search_meta_info(titles)
    print(result)