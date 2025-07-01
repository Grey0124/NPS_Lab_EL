# Deploying ARP Spoofing Detection Backend to Render

This guide will help you deploy your FastAPI backend to Render.

## Prerequisites

1. A GitHub account
2. A Render account (free tier available)
3. Your code pushed to a GitHub repository

## Step 1: Prepare Your Repository

Make sure your backend code is in a GitHub repository with the following structure:

```
arp_app/
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   ├── render.yaml
│   ├── Dockerfile
│   ├── .dockerignore
│   ├── services/
│   ├── api/
│   ├── models/
│   └── websocket/
```

## Step 2: Deploy to Render

### Option A: Using render.yaml (Recommended)

1. **Go to Render Dashboard**
   - Visit [render.com](https://render.com)
   - Sign in to your account

2. **Create New Web Service**
   - Click "New +" button
   - Select "Web Service"
   - Connect your GitHub repository

3. **Configure the Service**
   - **Name**: `arp-spoofing-detection-api`
   - **Environment**: `Python`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free (or choose paid plan)

4. **Environment Variables** (Optional)
   - `PYTHON_VERSION`: `3.11.0`
   - `PORT`: `8000` (Render will set this automatically)

5. **Deploy**
   - Click "Create Web Service"
   - Render will automatically build and deploy your application

### Option B: Using Docker

1. **Create New Web Service**
   - Click "New +" button
   - Select "Web Service"
   - Connect your GitHub repository

2. **Configure for Docker**
   - **Name**: `arp-spoofing-detection-api`
   - **Environment**: `Docker`
   - **Build Command**: Leave empty (uses Dockerfile)
   - **Start Command**: Leave empty (uses Dockerfile CMD)
   - **Plan**: Free (or choose paid plan)

3. **Deploy**
   - Click "Create Web Service"

## Step 3: Update Frontend Configuration

After deployment, update your frontend to use the new backend URL:

1. **Update API URL**
   ```typescript
   // In your frontend .env file or environment variables
   VITE_API_URL=https://your-app-name.onrender.com
   ```

2. **Update CORS Settings**
   - In `main.py`, update the `allow_origins` list with your frontend URL

## Step 4: Test Your Deployment

1. **Health Check**
   ```
   https://your-app-name.onrender.com/health
   ```

2. **API Documentation**
   ```
   https://your-app-name.onrender.com/docs
   ```

3. **Root Endpoint**
   ```
   https://your-app-name.onrender.com/
   ```

## Important Notes

### Limitations on Render Free Tier

1. **No Network Interface Access**: The ARP spoofing detection features that require network interface access won't work on Render's cloud environment. This is expected as cloud platforms don't provide direct network interface access.

2. **WebSocket Support**: WebSocket connections are supported on Render.

3. **Database**: The application uses SQLite which works fine on Render.

### What Will Work

- ✅ API endpoints
- ✅ Alert management
- ✅ Configuration management
- ✅ WebSocket connections
- ✅ Health checks
- ✅ Documentation

### What Won't Work

- ❌ ARP packet capture
- ❌ Network interface monitoring
- ❌ Real-time ARP spoofing detection

## Troubleshooting

### Common Issues

1. **Build Failures**
   - Check the build logs in Render dashboard
   - Ensure all dependencies are in `requirements.txt`
   - Verify Python version compatibility

2. **Runtime Errors**
   - Check the logs in Render dashboard
   - Ensure all required files are present
   - Verify environment variables

3. **CORS Issues**
   - Update the `allow_origins` list in `main.py`
   - Add your frontend domain to the allowed origins

### Logs and Monitoring

- View logs in the Render dashboard
- Monitor application health at `/health` endpoint
- Check build logs for deployment issues

## Environment Variables

You can set these in Render dashboard:

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Application port | `8000` |
| `PYTHON_VERSION` | Python version | `3.11.0` |

## Next Steps

1. **Deploy Frontend**: Consider deploying your React frontend to Vercel, Netlify, or Render
2. **Custom Domain**: Add a custom domain to your Render service
3. **Monitoring**: Set up monitoring and alerts
4. **Database**: Consider using a managed database service for production

## Support

- [Render Documentation](https://render.com/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Render Community](https://community.render.com/) 