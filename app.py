import featureSelection as FS

from flask import Flask, escape, url_for
import flask 

app = Flask(__name__)


@app.route('/')
def index():
    print(FS.greet())
    return 'index'


@app.route('/login')
def login():
    return 'login'

@app.route('/update_std_FS/<count_of_feature>/<department>/<datasetid>')
def update_std_FS(count_of_feature, department, datasetid):
    FS.update_FS(int(count_of_feature), './HR.csv', department, datasetid)
    response = flask.jsonify({'some': 'data'})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/user/<username>')
def profile(username):
    return '{}\'s profile'.format(escape(username))


with app.test_request_context():
    print(url_for('index'))
    print(url_for('login'))
    print(url_for('login', next='/'))
    print(url_for('profile', username='John Doe'))




