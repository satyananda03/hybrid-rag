import { useEffect, useState } from 'react'
import { toast } from 'sonner'
import { KnowledgeGraphViewer, getGraphWorkspaces, type GraphWorkspace } from '@/features/knowledge_graph'

export default function KnowledgeGraphPage() {
  const [workspaces, setWorkspaces] = useState<GraphWorkspace[]>([])
  const [activeWorkspace, setActiveWorkspace] = useState<GraphWorkspace | null>(null)
  const [loadFailed, setLoadFailed] = useState(false)

  useEffect(() => {
    void getGraphWorkspaces()
      .then((list) => {
        setWorkspaces(list)
        setActiveWorkspace((current) => current ?? list[0] ?? null)
      })
      .catch(() => {
        setLoadFailed(true)
        toast.error('Gagal memuat daftar workspace.')
      })
  }, [])

  return (
    <div className="flex h-screen flex-col overflow-hidden">
      <section className="relative min-h-0 flex-1">
        {/* `key` memaksa Sigma dan store-nya dibangun ulang saat ganti workspace;
            tanpa itu label, posisi node dan seleksi workspace lama ikut terbawa. */}
        {activeWorkspace && <KnowledgeGraphViewer key={activeWorkspace} workspace={activeWorkspace} />}

        {workspaces.length > 0 && (
          <select
            aria-label="Workspace"
            value={activeWorkspace ?? ''}
            onChange={(event) => setActiveWorkspace(event.target.value)}
            className="absolute top-4 right-4 z-30 rounded-lg border border-outline-variant/50 bg-surface-container-high px-3 py-2 text-sm text-on-surface shadow-lg outline-none focus:border-primary"
          >
            {workspaces.map((workspace) => (
              <option key={workspace} value={workspace}>{workspace}</option>
            ))}
          </select>
        )}

        {workspaces.length === 0 && (
          <div className="grid h-full place-items-center px-6 text-center text-sm text-on-surface-variant">
            {loadFailed
              ? 'Gagal memuat daftar workspace. Pastikan API berjalan, lalu muat ulang.'
              : 'Belum ada workspace yang berisi dokumen. Ingest dokumen dulu, lalu muat ulang.'}
          </div>
        )}
      </section>
    </div>
  )
}
