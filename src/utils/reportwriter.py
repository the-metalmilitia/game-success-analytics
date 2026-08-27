from pathlib import Path

class ReportWriter:
    def __init__(self, report_path: Path) -> None:
        self.path = report_path
        self._clean_report_file()


    #Wipes out the report text file and starts anew
    def _clean_report_file(self):
        with open(self.path, 'w', encoding='utf-8') as file:
            pass


    #appends next content to the report
    def append_to_report(self, content: str):
        with open(self.path, 'a', encoding='utf-8') as file:
            file.write(content)