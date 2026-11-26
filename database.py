from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class ScreeningRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(200), nullable=False)
    modality = db.Column(db.String(100), nullable=False)
    risk_level = db.Column(db.String(50), nullable=False)
    probability = db.Column(db.Float, nullable=False)
    ptosis_ratio = db.Column(db.Float, nullable=True)
    energy_level = db.Column(db.Float, nullable=True)
    timestamp = db.Column(db.DateTime, server_default=db.func.now())

    def __repr__(self):
        return f"<ScreeningRecord {self.filename} - {self.risk_level}>"

def init_db(app):
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///mg_project.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db.init_app(app)
    with app.app_context():
        db.create_all()