from sqlalchemy import Column,Integer,String,Boolean
from flaski.database import Base

class FishMaster(Base):
#テーブル名の設定
    __tablename__='fish_master'

    fish_name = Column(String,primary_key=True)
    poison = Column(Integer)
    poison_part = Column(String)
    wiki_url = Column(String)
    
    def __init__(self, fish_name=None,poison=None,poison_part=None,wiki_url=None):
        self.fish_name = fish_name
        self.poison = poison
        self.poison_part = poison_part
        self.wiki_url = wiki_url