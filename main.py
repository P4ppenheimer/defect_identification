from flask import Flask
from flask import send_file, make_response, request  # test for image posting
from flask_restful import Resource, Api
from apispec import APISpec
from marshmallow import Schema, fields
from apispec.ext.marshmallow import MarshmallowPlugin
from flask_apispec.extension import FlaskApiSpec
from flask_apispec.views import MethodResource
from webargs import fields, validate
from flask_apispec import marshal_with, doc, use_kwargs
import io
from PIL import Image

from prediction_function import *

# get model
model = get_model(ts = '2021-03-03 11_00_31')

# Flask stuff
app = Flask(__name__)  # Flask app instance initiated
api = Api(app)  # Flask restful wraps Flask app around it.

app.config.update({
    'APISPEC_SPEC': APISpec(
        title='Steel Defect Identification',
        version='v1',
        plugins=[MarshmallowPlugin()],
        openapi_version='2.0.0'
    ),
    'APISPEC_SWAGGER_URL': '/swagger/',  # URI to access API Doc JSON
    'APISPEC_SWAGGER_UI_URL': '/swagger-ui/',  # URI to access UI of API Doc
})
docs = FlaskApiSpec(app)


class AwesomeResponseSchema(Schema):
    message = fields.Str(default='Success')

class AwesomeRequestSchema(Schema):
    api_type = fields.String(required=True, description="API type of awesome API")


class AwesomeAPI(MethodResource, Resource):
    @doc(description='Get generated images.', tags=['The get statement'])
    @marshal_with(AwesomeResponseSchema)  # marshalling
    @app.route('/image/<image_id>')
    def get_image(image_id):

        print(image_id)

        img = get_prediction(idx=int(image_id), model=model)
        image = Image.fromarray((img * 255).astype(np.uint8))
        byteIO = io.BytesIO()
        image.save(byteIO, format='jpeg')
        byteIO.seek(0)

        response = send_file(byteIO, mimetype='image/jpeg')      
        response.cache_control.max_age = 0     

        return response

    docs.register(get_image)
        

if __name__ == '__main__':
    app.run(debug=True)

# backlog
# #  Restful way of creating APIs through Flask Restful
# class AwesomeAPI(MethodResource, Resource):

#     @doc(description='My First GET API.', tags=['Get statement'])
#     @marshal_with(AwesomeResponseSchema)  # marshalling
#     def get(self):
#         '''Get method represents a GET API method'''
        
#         idx = request.args.get('idx')

#         import random
#         randint = random.randint(0,2000)
#         randint = 127
#         randint = int(idx)

#         # get image as numpy array (w x h x c)
#         img = get_prediction(idx=randint, model=model)
        
#         # save image 
#         im = Image.fromarray((img * 255).astype(np.uint8))    
#         path = "./hacky_folder/hacky_plot.jpeg"
#         im.save(path)

#         response = send_file(path, mimetype='image/jpeg')      
#         response.cache_control.max_age = 0                      

#         return response
        
#     @doc(description='My First GET Awesome API.', tags=['Post statement'])
#     @use_kwargs(AwesomeRequestSchema, location=('json'))
#     @marshal_with(AwesomeResponseSchema)  # marshalling
#     def post(self, test):
#         '''Post method represents a PUT API method'''      
#         return {'Post statement' : 'Hello World'}

# api.add_resource(AwesomeAPI, '/Get segmented image')
# docs.register(AwesomeAPI)