import json
import os
import ast
from typing import List, Dict, Optional
import logging
from . import config

logger = logging.getLogger(__name__)

class AlphaLibrary:
    def __init__(self, db_path: str = config.DB_PATH):
        self.db_path = db_path
        self.genes: List[Dict] = self._load_db()

    def _load_db(self) -> List[Dict]:
        if not os.path.exists(self.db_path):
            logger.info("Database not found, starting fresh.")
            return []
        try:
            with open(self.db_path, 'r') as f:
                logger.info(f"Loading database from {self.db_path}")
                return json.load(f)
        except json.JSONDecodeError:
            logger.error("Database file corrupted. Starting fresh.")
            return []

    def _save_db(self):
        with open(self.db_path, 'w') as f:
            json.dump(self.genes, f, indent=2)

    def save_alpha(self, state: Dict) -> str:
        """Saves a successful strategy to the gene pool."""
        import uuid
        from datetime import datetime
        
        # Use existing ID or generate new if not present (though Architect usually handles ID naming, UUID is safer for DB)
        
        # Extract metrics
        metrics = state.get("backtest_metrics", {})
        
        entry = {
            "id": str(uuid.uuid4())[:8],
            "generation": state.get("generation", 0),
            "concept": state.get("strategy_concept", ""),
            "code": state.get("python_code", ""),
            "metrics": metrics,
            "parents": state.get("parent_ids", [])
        }
        
        self.genes.append(entry)
        logger.info(f"Alpha {entry['id']} SAVED to library. Sharpe: {metrics.get('Sharpe')}")

        # Expert Code to File (Preserving existing functionality)
        try:
            # Check if user provided specific path
            custom_path = state.get("saveto")
            
            if custom_path:
                filename = os.path.abspath(os.path.expanduser(custom_path))
                # Ensure user's directory exists (if path is "foo.py", dirname is empty/current dir)
                dir_name = os.path.dirname(filename)
                if dir_name:
                    os.makedirs(dir_name, exist_ok=True)
            else:
                os.makedirs(config.SUCCESSFUL_ALGOS_DIR, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Use concept name if available in concept text, else ID
                safe_name = f"Gen{entry['generation']}_{entry['id']}"
                filename = os.path.join(config.SUCCESSFUL_ALGOS_DIR, f"{safe_name}_{timestamp}.py")
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(entry['code'])
            logger.info(f"Alpha Code exported to: {filename}")
        except Exception as e:
            logger.error(f"Failed to export alpha code to file: {e}")

        # Keep only top performers if pool gets too big
        self._save_db()
        return entry["id"]



    def is_duplicate(self, code: str) -> bool:
        """Uses AST to compare code structure and prevent identical clones."""
        try:
            new_tree = ast.parse(code)
            new_dump = ast.dump(new_tree)
            
            for existing_gene in self.genes:
                existing_code = existing_gene.get('code', "")
                try:
                    existing_tree = ast.parse(existing_code)
                    if ast.dump(existing_tree) == new_dump:
                        return True
                except SyntaxError:
                    continue 
        except SyntaxError:
            return False 
            
        return False
        
    def is_empty(self) -> bool:
        return len(self.genes) == 0

    def clear(self):
        """Clears the library (useful for testing)"""
        self.genes = []
        self._save_db()
