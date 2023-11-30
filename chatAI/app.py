from flask import Flask, render_template, request, jsonify
from chatty import chat
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # this line enables CORS(cross origin recourse sharing)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chatbot', methods=['POST'])
def chatbot_api():
    if request.method == 'POST':
        
        user_message = request.json.get('message')
        # print(user_message)
        chatbot_response = chat(user_message)
        
        return jsonify({"message": chatbot_response})

if __name__ == '__main__':
    app.run(debug=True)