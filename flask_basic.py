from flask import Flask
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}

class This_is_a_test(Resource):
    def get(self):
        return {'hello': 'you'}

api.add_resource(HelloWorld, '/')
api.add_resource(This_is_a_test, '/test')

if __name__ == '__main__':
    app.run(debug=True)
