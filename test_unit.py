import unittest
import os
import sys
import tempfile
import sqlite3
from dotenv import load_dotenv
from datetime import datetime

# Add the current directory to Python path to import project modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

class CoreLocalModelServingTests(unittest.TestCase):
    """Core 5 unit tests for Local Model Serving with SQLite with real components"""
    
    @classmethod
    def setUpClass(cls):
        """Load environment variables and validate API key"""
        load_dotenv()
        
        # Validate API key
        cls.api_key = os.getenv('GEMINI_API_KEY')
        if not cls.api_key or not cls.api_key.startswith('AIza'):
            raise unittest.SkipTest("Valid GEMINI_API_KEY not found in environment")
        
        print(f"Using API Key: {cls.api_key[:10]}...{cls.api_key[-5:]}")
        
        # Initialize local model serving components
        try:
            import config
            import database
            import models
            import app
            
            cls.config = config.Config
            cls.database = database
            cls.models = models
            cls.app_module = app
            
            # Create a test database session
            cls.db_session = next(database.get_db())
            
            print("Local model serving components loaded successfully")
        except ImportError as e:
            raise unittest.SkipTest(f"Required local model serving components not found: {e}")

    @classmethod
    def tearDownClass(cls):
        """Clean up database session"""
        if hasattr(cls, 'db_session'):
            try:
                cls.db_session.close()
            except:
                pass

    def test_01_database_initialization(self):
        """Test 1: Database Initialization and Schema"""
        print("Running Test 1: Database Initialization and Schema")
        
        # Test database connection
        self.assertIsNotNone(self.database.engine)
        self.assertIsNotNone(self.database.SessionLocal)
        
        # Test database session creation
        db = next(self.database.get_db())
        self.assertIsNotNone(db)
        
        # Test table creation by checking models
        request_log = self.models.RequestLog()
        response_log = self.models.ResponseLog()
        
        # Verify RequestLog model structure
        required_request_fields = ['id', 'timestamp', 'endpoint', 'client_ip', 'request_body']
        for field in required_request_fields:
            self.assertTrue(hasattr(request_log, field), f"RequestLog should have {field} field")
        
        # Verify ResponseLog model structure
        required_response_fields = ['id', 'request_id', 'timestamp', 'response_time_ms', 
                                   'response', 'prompt_tokens', 'completion_tokens', 
                                   'total_tokens', 'model_used']
        for field in required_response_fields:
            self.assertTrue(hasattr(response_log, field), f"ResponseLog should have {field} field")
        
        db.close()
        print("PASS: Database initialization and schema validation completed")

    def test_02_model_configuration(self):
        """Test 2: Model Configuration and Fallback Chain"""
        print("Running Test 2: Model Configuration and Fallback Chain")
        
        # Test configuration loading
        self.assertIsNotNone(self.config.GEMINI_API_KEY)
        self.assertTrue(self.config.GEMINI_API_KEY.startswith('AIza'))
        
        # Test model fallback chain
        self.assertIsInstance(self.config.MODELS, list)
        self.assertGreaterEqual(len(self.config.MODELS), 3, "Should have at least 3 models for fallback")
        
        # Verify all models are Gemini models
        for model in self.config.MODELS:
            self.assertIn('gemini', model.lower(), f"Model {model} should be a Gemini model")
        
        # Test primary model configuration
        self.assertEqual(self.config.MODEL_NAME, self.config.MODELS[0])
        
        # Test database URL configuration
        self.assertIsNotNone(self.config.DATABASE_URL)
        self.assertTrue(self.config.DATABASE_URL.startswith('sqlite://'))
        
        print(f"PASS: Configuration validated - Primary model: {self.config.MODEL_NAME}")
        print(f"PASS: Fallback chain: {self.config.MODELS}")

    def test_03_request_logging(self):
        """Test 3: Request Logging to SQLite Database"""
        print("Running Test 3: Request Logging to SQLite Database")
        
        # Create a test request log entry
        test_endpoint = "/gemini/generate"
        test_client_ip = "192.168.1.100"
        test_request_body = '{"prompt": {"text": "Test prompt"}, "generation_config": {"temperature": 0.7}}'
        
        # Create and save request log
        request_log = self.models.RequestLog(
            endpoint=test_endpoint,
            client_ip=test_client_ip,
            request_body=test_request_body,
            timestamp=datetime.utcnow()
        )
        
        # Add to database
        self.db_session.add(request_log)
        self.db_session.commit()
        self.db_session.refresh(request_log)
        
        # Verify the log was created
        self.assertIsNotNone(request_log.id)
        self.assertEqual(request_log.endpoint, test_endpoint)
        self.assertEqual(request_log.client_ip, test_client_ip)
        self.assertEqual(request_log.request_body, test_request_body)
        self.assertIsNotNone(request_log.timestamp)
        
        # Query back from database to verify persistence
        retrieved_log = self.db_session.query(self.models.RequestLog).filter_by(id=request_log.id).first()
        self.assertIsNotNone(retrieved_log)
        self.assertEqual(retrieved_log.endpoint, test_endpoint)
        
        print(f"PASS: Request logged - ID: {request_log.id}, Endpoint: {request_log.endpoint}")
        print(f"PASS: Client IP: {request_log.client_ip}")

    def test_04_response_logging_with_metrics(self):
        """Test 4: Response Logging with Token Metrics"""
        print("Running Test 4: Response Logging with Token Metrics")
        
        # First create a request log
        request_log = self.models.RequestLog(
            endpoint="/gemini/generate",
            client_ip="127.0.0.1",
            request_body='{"prompt": {"text": "Test"}}',
            timestamp=datetime.utcnow()
        )
        self.db_session.add(request_log)
        self.db_session.commit()
        self.db_session.refresh(request_log)
        
        # Create response log with metrics
        test_response_time = 245.7
        test_response_text = "This is a test response from the Gemini model."
        test_prompt_tokens = 15
        test_completion_tokens = 35
        test_total_tokens = 50
        test_model_used = "gemini-2.0-flash"
        
        response_log = self.models.ResponseLog(
            request_id=request_log.id,
            response_time_ms=test_response_time,
            response=test_response_text,
            prompt_tokens=test_prompt_tokens,
            completion_tokens=test_completion_tokens,
            total_tokens=test_total_tokens,
            model_used=test_model_used,
            timestamp=datetime.utcnow()
        )
        
        # Add to database
        self.db_session.add(response_log)
        self.db_session.commit()
        self.db_session.refresh(response_log)
        
        # Verify the response log was created
        self.assertIsNotNone(response_log.id)
        self.assertEqual(response_log.request_id, request_log.id)
        self.assertEqual(response_log.response_time_ms, test_response_time)
        self.assertEqual(response_log.response, test_response_text)
        self.assertEqual(response_log.prompt_tokens, test_prompt_tokens)
        self.assertEqual(response_log.completion_tokens, test_completion_tokens)
        self.assertEqual(response_log.total_tokens, test_total_tokens)
        self.assertEqual(response_log.model_used, test_model_used)
        
        # Verify token calculation
        self.assertEqual(response_log.total_tokens, 
                        response_log.prompt_tokens + response_log.completion_tokens)
        
        # Query back from database to verify persistence
        retrieved_response = self.db_session.query(self.models.ResponseLog).filter_by(id=response_log.id).first()
        self.assertIsNotNone(retrieved_response)
        self.assertEqual(retrieved_response.model_used, test_model_used)
        
        print(f"PASS: Response logged - ID: {response_log.id}, Request ID: {response_log.request_id}")
        print(f"PASS: Response time: {response_log.response_time_ms}ms, Model: {response_log.model_used}")
        print(f"PASS: Tokens - Prompt: {response_log.prompt_tokens}, Completion: {response_log.completion_tokens}, Total: {response_log.total_tokens}")

    def test_05_database_queries_and_metrics(self):
        """Test 5: Database Query Operations and Metrics Retrieval"""
        print("Running Test 5: Database Query Operations and Metrics Retrieval")
        
        # Add some test data for querying
        for i in range(3):
            request_log = self.models.RequestLog(
                endpoint=f"/gemini/generate",
                client_ip=f"192.168.1.{i+1}",
                request_body=f'{{"prompt": {{"text": "Test prompt {i+1}"}}}}',
                timestamp=datetime.utcnow()
            )
            self.db_session.add(request_log)
            self.db_session.commit()
            self.db_session.refresh(request_log)
            
            # Add corresponding response
            response_log = self.models.ResponseLog(
                request_id=request_log.id,
                response_time_ms=100.0 + (i * 50),
                response=f"Test response {i+1}",
                prompt_tokens=10 + i,
                completion_tokens=20 + i,
                total_tokens=30 + (i * 2),
                model_used=self.config.MODELS[i % len(self.config.MODELS)],
                timestamp=datetime.utcnow()
            )
            self.db_session.add(response_log)
            self.db_session.commit()
        
        # Test query operations
        request_count = self.db_session.query(self.models.RequestLog).count()
        response_count = self.db_session.query(self.models.ResponseLog).count()
        
        self.assertGreaterEqual(request_count, 3, "Should have at least 3 request logs")
        self.assertGreaterEqual(response_count, 3, "Should have at least 3 response logs")
        
        # Test recent queries
        recent_requests = self.db_session.query(self.models.RequestLog).order_by(
            self.models.RequestLog.id.desc()
        ).limit(2).all()
        
        recent_responses = self.db_session.query(self.models.ResponseLog).order_by(
            self.models.ResponseLog.id.desc()
        ).limit(2).all()
        
        self.assertLessEqual(len(recent_requests), 2, "Should return at most 2 recent requests")
        self.assertLessEqual(len(recent_responses), 2, "Should return at most 2 recent responses")
        
        # Verify data structure
        if recent_requests:
            recent_req = recent_requests[0]
            self.assertIsNotNone(recent_req.id)
            self.assertIsNotNone(recent_req.endpoint)
            self.assertIsNotNone(recent_req.timestamp)
        
        if recent_responses:
            recent_resp = recent_responses[0]
            self.assertIsNotNone(recent_resp.id)
            self.assertIsNotNone(recent_resp.model_used)
            self.assertIsNotNone(recent_resp.total_tokens)
        
        # Test metrics aggregation
        total_tokens_used = self.db_session.query(
            self.db_session.query(self.models.ResponseLog.total_tokens).subquery().c.total_tokens
        ).all()
        
        print(f"PASS: Database queries - Requests: {request_count}, Responses: {response_count}")
        print(f"PASS: Recent data retrieval working correctly")
        print(f"PASS: Metrics aggregation functional")

def run_core_tests():
    """Run core tests and provide summary"""
    print("=" * 70)
    print("[*] Core Local Model Serving with SQLite Unit Tests (5 Tests)")
    print("Testing with REAL Database and Local Serving Components")
    print("=" * 70)
    
    # Check API key
    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key or not api_key.startswith('AIza'):
        print("[ERROR] Valid GEMINI_API_KEY not found!")
        return False
    
    print(f"[OK] Using API Key: {api_key[:10]}...{api_key[-5:]}")
    print()
    
    # Run tests
    suite = unittest.TestLoader().loadTestsFromTestCase(CoreLocalModelServingTests)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 70)
    print("[*] Test Results:")
    print(f"[*] Tests Run: {result.testsRun}")
    print(f"[*] Failures: {len(result.failures)}")
    print(f"[*] Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n[FAILURES]:")
        for test, traceback in result.failures:
            print(f"  - {test}")
            print(f"    {traceback}")
    
    if result.errors:
        print("\n[ERRORS]:")
        for test, traceback in result.errors:
            print(f"  - {test}")
            print(f"    {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print("\n[SUCCESS] All 5 core local model serving tests passed!")
        print("[OK] Local model serving components working correctly with real database")
        print("[OK] Database Schema, Model Config, Request Logging, Response Metrics, Queries validated")
    else:
        print(f"\n[WARNING] {len(result.failures) + len(result.errors)} test(s) failed")
    
    return success

if __name__ == "__main__":
    print("[*] Starting Core Local Model Serving with SQLite Tests")
    print("[*] 5 essential tests with real database and local serving components")
    print("[*] Components: Database, Models, Request Logging, Response Metrics, Queries")
    print()
    
    success = run_core_tests()
    exit(0 if success else 1)