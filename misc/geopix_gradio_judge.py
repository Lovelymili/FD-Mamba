import os
import re
import tempfile
from typing import Optional, Tuple
from PIL import Image
from gradio_client import Client


class GeoPixGradioJudge:
    """
    调用本地 GeoPix gradio demo (api_name=/inference) 做 YES/NO 裁判。
    """

    def __init__(self, server_url: str = "http://127.0.0.1:7860/", api_name: str = "/inference", timeout: int = 120):
        self.client = Client(server_url)
        self.api_name = api_name
        self.timeout = timeout

    @staticmethod
    def _parse_yes_no(text: str) -> Optional[bool]:
        if text is None:
            return None
        t = text.strip().upper()
        
        if re.search(r"\bYES\b", t):
            return True
        if re.search(r"\bNO\b", t):
            return False
        return None

    def judge_keep(self, pair_img: Image.Image, question: str, default_keep: bool = True) -> Tuple[bool, str]:
        """
        return: (keep?, raw_text_output)
        """
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_path = f.name
        try:
            pair_img.save(tmp_path)

            
            text_output, _image_output = self.client.predict(
                "Visual Question Answering",  
                question,                     
                tmp_path,                     
                api_name=self.api_name,
            )

            parsed = self._parse_yes_no(str(text_output))
            if parsed is None:
                return default_keep, str(text_output)
            return parsed, str(text_output)

        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
