# Frontend Deployment Guide

Your frontend is now configured to connect to the deployed backend at `https://nps-lab-el.onrender.com`.

## Current Configuration

- **Backend URL**: `https://nps-lab-el.onrender.com`
- **API Base URL**: `https://nps-lab-el.onrender.com/api/v1`
- **WebSocket URL**: `wss://nps-lab-el.onrender.com/ws`

## Environment Variables

The following environment variables are configured:

```env
VITE_API_URL=https://nps-lab-el.onrender.com
VITE_FIREBASE_API_KEY=AIzaSyCKr3i38gFBBagMVjMmvTHH4hu7mkFKzpw
VITE_FIREBASE_AUTH_DOMAIN=arp-spoof-8d356.firebaseapp.com
VITE_FIREBASE_PROJECT_ID=arp-spoof-8d356
VITE_FIREBASE_STORAGE_BUCKET=arp-spoof-8d356.firebasestorage.app
VITE_FIREBASE_MESSAGING_SENDER_ID=439105494123
VITE_FIREBASE_APP_ID=1:439105494123:web:a887185d211f1d9cc2252e
VITE_FIREBASE_MEASUREMENT_ID=G-25QPXM299C
```

## Deploy to Vercel

### Option 1: Using Vercel CLI

1. **Install Vercel CLI**:
   ```bash
   npm install -g vercel
   ```

2. **Login to Vercel**:
   ```bash
   vercel login
   ```

3. **Deploy**:
   ```bash
   cd arp_app/frontend/arp_spoof
   vercel
   ```

4. **Configure Environment Variables**:
   - Go to your Vercel dashboard
   - Select your project
   - Go to Settings → Environment Variables
   - Add all the environment variables from above

### Option 2: Using Vercel Dashboard

1. **Connect GitHub Repository**:
   - Go to [vercel.com](https://vercel.com)
   - Click "New Project"
   - Import your GitHub repository

2. **Configure Project**:
   - **Framework Preset**: Vite
   - **Root Directory**: `arp_app/frontend/arp_spoof`
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`

3. **Environment Variables**:
   - Add all environment variables from the list above

4. **Deploy**:
   - Click "Deploy"

## Deploy to Netlify

1. **Connect Repository**:
   - Go to [netlify.com](https://netlify.com)
   - Click "New site from Git"
   - Connect your GitHub repository

2. **Configure Build Settings**:
   - **Base directory**: `arp_app/frontend/arp_spoof`
   - **Build command**: `npm run build`
   - **Publish directory**: `dist`

3. **Environment Variables**:
   - Go to Site settings → Environment variables
   - Add all environment variables from the list above

4. **Deploy**:
   - Click "Deploy site"

## Deploy to Render

1. **Create Static Site**:
   - Go to [render.com](https://render.com)
   - Click "New +" → "Static Site"
   - Connect your GitHub repository

2. **Configure**:
   - **Name**: `arp-spoofing-frontend`
   - **Root Directory**: `arp_app/frontend/arp_spoof`
   - **Build Command**: `npm run build`
   - **Publish Directory**: `dist`

3. **Environment Variables**:
   - Add all environment variables from the list above

4. **Deploy**:
   - Click "Create Static Site"

## Testing the Connection

Before deploying, you can test the backend connection:

```bash
cd arp_app/frontend/arp_spoof
node test-backend-connection.js
```

This will test:
- ✅ Ping endpoint
- ✅ Health check
- ✅ API v1 health
- ✅ CORS configuration

## Post-Deployment

After deployment:

1. **Update CORS Settings** (if needed):
   - If you deploy to a different domain, update the CORS settings in the backend
   - Add your frontend URL to the `allow_origins` list in `main.py`

2. **Test the Application**:
   - Navigate to your deployed frontend
   - Check if it connects to the backend
   - Test the WebSocket connection
   - Verify all features work correctly

## Troubleshooting

### Common Issues

1. **CORS Errors**:
   - Ensure your frontend domain is in the backend's CORS allow list
   - Check that the backend is accessible

2. **WebSocket Connection Failed**:
   - Verify the WebSocket URL is correct
   - Check if your hosting platform supports WebSockets

3. **Environment Variables Not Working**:
   - Ensure all variables have the `VITE_` prefix
   - Rebuild the application after adding variables

4. **Build Failures**:
   - Check that all dependencies are installed
   - Verify TypeScript compilation passes

### Support

- [Vercel Documentation](https://vercel.com/docs)
- [Netlify Documentation](https://docs.netlify.com/)
- [Render Documentation](https://render.com/docs) 