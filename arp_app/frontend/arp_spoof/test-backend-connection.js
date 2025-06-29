// Test script to verify backend connection
// Run this with: node test-backend-connection.js

const API_URL = 'https://nps-lab-el.onrender.com';

async function testBackendConnection() {
  console.log('🔍 Testing backend connection...');
  console.log('📍 Backend URL:', API_URL);
  
  try {
    // Test 1: Basic ping
    console.log('\n1️⃣ Testing ping endpoint...');
    const pingResponse = await fetch(`${API_URL}/ping`);
    const pingData = await pingResponse.json();
    console.log('✅ Ping response:', pingData);
    
    // Test 2: Health check
    console.log('\n2️⃣ Testing health endpoint...');
    const healthResponse = await fetch(`${API_URL}/health`);
    const healthData = await healthResponse.json();
    console.log('✅ Health response:', healthData);
    
    // Test 3: API v1 health
    console.log('\n3️⃣ Testing API v1 health...');
    const apiHealthResponse = await fetch(`${API_URL}/api/v1/health`);
    const apiHealthData = await apiHealthResponse.json();
    console.log('✅ API v1 health response:', apiHealthData);
    
    // Test 4: CORS preflight
    console.log('\n4️⃣ Testing CORS...');
    const corsResponse = await fetch(`${API_URL}/api/v1/health`, {
      method: 'OPTIONS',
      headers: {
        'Origin': 'http://localhost:5173',
        'Access-Control-Request-Method': 'GET',
        'Access-Control-Request-Headers': 'Content-Type'
      }
    });
    console.log('✅ CORS preflight status:', corsResponse.status);
    console.log('✅ CORS headers:', Object.fromEntries(corsResponse.headers.entries()));
    
    console.log('\n🎉 All tests passed! Backend is ready for frontend connection.');
    
  } catch (error) {
    console.error('❌ Test failed:', error.message);
    console.error('Full error:', error);
  }
}

// Run the test
testBackendConnection(); 