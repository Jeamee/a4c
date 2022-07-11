class Notebook():
    def __init__(self, notebook_id, codes, markdowns, markdown_ids=None):
        self.id = notebook_id
        self.codes = codes
        self.markdowns = markdowns
        self.markdown_ids = markdown_ids
        
        self.labels = range(len(self.markdowns))
