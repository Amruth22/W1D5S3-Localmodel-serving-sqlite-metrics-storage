import time
import logging
import traceback
from flask import Flask, request
from flask_restx import Api, Resource, fields
import google.generativeai as genai
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import text

from config import Config
from database import get_db, engine
from models import Base, RequestLog, ResponseLog
from init_db import init_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize database and create tables if needed
try:
    init_db()
    logger.info("Database initialization complete")
except Exception as db_init_error:
    logger.error(f"Database initialization error: {str(db_init_error)}")
    logger.error(traceback.format_exc())

# Initialize Gemini client
genai.configure(api_key=Config.GEMINI_API_KEY)

# Initialize Flask app
app = Flask(__name__)
api = Api(app, 
    version='1.0',
    title='Gemini Proxy Server',
    description='Local model serving with SQLite metrics storage',
    doc='/docs'
)

# Define API models
ns = api.namespace('gemini', description='Gemini API')

# Request model
prompt_model = api.model('Prompt', {
    'text': fields.String(required=True, description='The text prompt')
})

generation_config = api.model('GenerationConfig', {
    'temperature': fields.Float(description='Temperature (0.0-1.0)'),
    'max_output_tokens': fields.Integer(description='Max tokens to generate'),
    'top_p': fields.Float(description='Top-p sampling'),
    'top_k': fields.Integer(description='Top-k sampling')
})

request_model = api.model('GenerateRequest', {
    'prompt': fields.Nested(prompt_model, required=True),
    'generation_config': fields.Nested(generation_config, required=False)
})

# Response model
response_model = api.model('GenerateResponse', {
    'text': fields.String(description='Generated text'),
    'metrics': fields.Raw(description='Usage metrics'),
    'model': fields.String(description='Model used for generation')
})

@ns.route('/generate')
class Generate(Resource):
    @ns.expect(request_model)
    @ns.marshal_with(response_model)
    def post(self):
        """Generate text using Gemini API"""
        start_time = time.time()
        
        # Get database session with error handling
        db = None
        try:
            # Safely get database connection
            try:
                db = next(get_db())
                logger.info("Successfully connected to database")
            except Exception as db_error:
                logger.error(f"Database connection error: {str(db_error)}")
                logger.error(traceback.format_exc())
                return {'message': f'Database connection error: {str(db_error)}'}, 500
            # Parse request
            data = request.json
            prompt_text = data['prompt']['text']
            
            # Log request with error handling
            try:
                client_ip = request.remote_addr
                endpoint = request.endpoint
                request_log = RequestLog(
                    endpoint=endpoint,
                    client_ip=client_ip,
                    request_body=str(data) # Storing as string for simplicity
                )
                db.add(request_log)
                db.commit()
                db.refresh(request_log)
                logger.info(f"Logged request with ID: {request_log.id}")
            except SQLAlchemyError as db_error:
                db.rollback()
                logger.error(f"Database error logging request: {str(db_error)}")
                logger.error(traceback.format_exc())
                # Continue without DB logging if there's an error
                request_log = None
            
            # Configure generation parameters
            gen_config = {}
            if 'generation_config' in data:
                cfg = data['generation_config']
                if 'temperature' in cfg:
                    gen_config['temperature'] = cfg['temperature']
                if 'max_output_tokens' in cfg:
                    gen_config['max_output_tokens'] = cfg['max_output_tokens']
                if 'top_p' in cfg:
                    gen_config['top_p'] = cfg['top_p']
                if 'top_k' in cfg:
                    gen_config['top_k'] = cfg['top_k']
            
            # Call Gemini API with fallback mechanism
            response = None
            last_error = None
            successful_model = None
            
            # Try to get available models first
            try:
                available_models = genai.list_models()
                logger.info(f"Available models: {[model.name for model in available_models]}")
            except Exception as model_list_error:
                logger.warning(f"Could not list available models: {str(model_list_error)}")
                available_models = []
            
            # Try each model in our fallback chain
            for model_name in Config.MODELS:
                try:
                    logger.info(f"Attempting to use model: {model_name}")
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content(prompt_text, generation_config=gen_config)
                    logger.info(f"Successfully generated content with model {model_name}")
                    successful_model = model_name
                    break
                except Exception as e:
                    error_message = str(e)
                    logger.warning(f"Error using {model_name}: {error_message}")
                    last_error = e
                    
                    # If it's not a quota error (429), no need to try other models
                    if "429" not in error_message and "quota" not in error_message.lower():
                        logger.error(f"Non-quota error encountered: {error_message}")
                        raise
            
            # If no model worked
            if not response:
                logger.error("All models failed")
                raise last_error
            
            # Calculate response time
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            
            # Extract metrics
            metrics = {}
            # Handle different response structures for different model versions
            try:
                # Try to get usage metadata - structure might vary between model versions
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    # Gemini 1.0/1.5 structure
                    prompt_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
                    completion_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)
                    total_tokens = getattr(response.usage_metadata, 'total_token_count', 0)
                    metrics = {
                        'prompt_tokens': prompt_tokens,
                        'completion_tokens': completion_tokens,
                        'total_tokens': total_tokens
                    }
                elif hasattr(response, 'usage') and response.usage:
                    # Alternative structure for newer models
                    metrics = {
                        'prompt_tokens': getattr(response.usage, 'input_tokens', 0),
                        'completion_tokens': getattr(response.usage, 'output_tokens', 0),
                        'total_tokens': getattr(response.usage, 'total_tokens', 0)
                    }
                elif hasattr(response, '_response') and hasattr(response._response, 'usage_metadata'):
                    # Access via protected _response attribute if available
                    usage = response._response.usage_metadata
                    metrics = {
                        'prompt_tokens': getattr(usage, 'prompt_token_count', 0),
                        'completion_tokens': getattr(usage, 'candidates_token_count', 0),
                        'total_tokens': getattr(usage, 'total_token_count', 0)
                    }
                else:
                    # Default to empty metrics if we can't find usage data
                    logger.warning(f"Could not find usage metadata in response: {type(response)}")
                    metrics = {
                        'prompt_tokens': 0,
                        'completion_tokens': 0,
                        'total_tokens': 0
                    }
                    
                # Log the response structure for debugging
                logger.info(f"Response attributes: {dir(response)}")
            except Exception as metrics_error:
                logger.warning(f"Error extracting metrics: {str(metrics_error)}")
                metrics = {
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'total_tokens': 0
                }
            
            # Extract response text
            response_text = ""
            try:
                # Try different possible response structures
                if hasattr(response, 'text'):
                    # Direct text attribute
                    response_text = response.text
                elif hasattr(response, 'candidates') and response.candidates:
                    # Gemini 1.0/1.5 structure with candidates
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, 'text'):
                            response_text += part.text
                elif hasattr(response, 'parts'):
                    # Direct parts attribute
                    for part in response.parts:
                        if hasattr(part, 'text'):
                            response_text += part.text
                elif hasattr(response, 'content') and hasattr(response.content, 'parts'):
                    # Content with parts structure
                    for part in response.content.parts:
                        if hasattr(part, 'text'):
                            response_text += part.text
                else:
                    # Last resort: convert the whole response to string
                    response_text = str(response)
                    # Clean up if it looks like a representation rather than content
                    if response_text.startswith('<') and '>' in response_text[:100]:
                        logger.warning("Could not extract text properly, using string representation")
                
                if not response_text:
                    logger.warning(f"Could not extract text from response: {type(response)}")
                    logger.info(f"Response structure: {dir(response)}")
                    response_text = "[No text content could be extracted]"
            except Exception as text_error:
                logger.warning(f"Error extracting text: {str(text_error)}")
                response_text = f"[Error extracting text: {str(text_error)}]"
            
            # Log response with error handling
            try:
                if request_log is not None and db is not None:
                    response_log = ResponseLog(
                        request_id=request_log.id,
                        response_time_ms=response_time_ms,
                        response=response_text,
                        prompt_tokens=metrics.get('prompt_tokens', 0),
                        completion_tokens=metrics.get('completion_tokens', 0),
                        total_tokens=metrics.get('total_tokens', 0),
                        model_used=successful_model or 'unknown'
                    )
                    db.add(response_log)
                    db.commit()
                    logger.info(f"Logged response with ID: {response_log.id}")
            except SQLAlchemyError as db_error:
                if db is not None:
                    db.rollback()
                logger.error(f"Database error logging response: {str(db_error)}")
                logger.error(traceback.format_exc())
                # Continue without DB logging if there's an error
            
            # Return response
            return {
                'text': response_text,
                'metrics': metrics,
                'model': successful_model or 'unknown'
            }
            
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            logger.error(traceback.format_exc())
            if db is not None:
                try:
                    db.rollback()
                except Exception:
                    pass
            api.abort(500, f"Server error: {str(e)}")
        finally:
            if db is not None:
                try:
                    db.close()
                except Exception as close_error:
                    logger.error(f"Error closing database connection: {str(close_error)}")
                    pass

# Health check endpoint
@app.route('/health')
def health():
    """Health check endpoint"""
    return {'status': 'healthy'}

# Model info endpoint
@app.route('/models')
def list_models():
    """List available models"""
    try:
        models = genai.list_models()
        return {'models': [{'name': model.name, 'display_name': model.display_name} for model in models]}
    except Exception as e:
        return {'error': str(e)}

# Database check endpoint
@app.route('/db-check')
def db_check():
    """Check database connection and schema"""
    try:
        db = next(get_db())
        try:
            # Check request_logs table
            request_count = db.query(RequestLog).count()
            
            # Check response_logs table
            response_count = db.query(ResponseLog).count()
            
            # Get the most recent entries
            recent_requests = db.query(RequestLog).order_by(RequestLog.id.desc()).limit(3).all()
            recent_responses = db.query(ResponseLog).order_by(ResponseLog.id.desc()).limit(3).all()
            
            # Format data for response
            requests = [{
                'id': req.id,
                'timestamp': req.timestamp.isoformat() if hasattr(req, 'timestamp') else None,
                'endpoint': req.endpoint if hasattr(req, 'endpoint') else None
            } for req in recent_requests]
            
            responses = [{
                'id': resp.id,
                'request_id': resp.request_id if hasattr(resp, 'request_id') else None,
                'timestamp': resp.timestamp.isoformat() if hasattr(resp, 'timestamp') else None,
                'model_used': resp.model_used if hasattr(resp, 'model_used') else None
            } for resp in recent_responses]
            
            return {
                'status': 'Database connection successful',
                'tables': {
                    'request_logs': {
                        'count': request_count,
                        'recent': requests
                    },
                    'response_logs': {
                        'count': response_count,
                        'recent': responses
                    }
                }
            }
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Database check error: {str(e)}")
        logger.error(traceback.format_exc())
        return {'status': 'error', 'message': str(e)}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)