from flask import Flask

app = Flask(__name__)

@app.route('/test')
def test():
    return 'Hello from Docker API!'

if __name__ == '__main__':
    # Make the server publicly available
    app.run(host='0.0.0.0', port=5000)