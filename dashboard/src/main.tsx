import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { Toaster } from 'sonner'
import '@/styles/globals.css'
import KnowledgeGraphPage from '@/pages/KnowledgeGraphPage'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <KnowledgeGraphPage />
    <Toaster richColors position="top-center" />
  </StrictMode>,
)
