import { useMemo, type ReactNode } from 'react'
import { GitBranchPlus, Scissors, X } from 'lucide-react'
import { nodeDisplayName } from '@/features/knowledge_graph/graph'
import { selectedElement, useKnowledgeGraphStore } from '@/features/knowledge_graph/store'

// Panel ini read-only: backend sengaja tidak punya rute tulis graph (edit entity,
// edit relation, merge). Expand dan Prune tetap ada — Expand memuat subgraph lewat
// GET /graphs, Prune hanya membuang node dari tampilan.
interface GraphPropertiesProps { onExpand: (nodeId: string) => void }

export function GraphProperties({ onExpand }: GraphPropertiesProps) {
  const rawGraph = useKnowledgeGraphStore((state) => state.rawGraph)
  const selectedNode = useKnowledgeGraphStore((state) => state.selectedNode)
  const selectedEdge = useKnowledgeGraphStore((state) => state.selectedEdge)
  const focusedNode = useKnowledgeGraphStore((state) => state.focusedNode)
  const focusedEdge = useKnowledgeGraphStore((state) => state.focusedEdge)
  const pruneNode = useKnowledgeGraphStore((state) => state.pruneNode)
  // Priority matches graph_ui/src/components/graph/PropertiesView.tsx: a
  // hovered (focused) node/edge takes over the panel from the selected one.
  const activeNode = focusedNode ?? selectedNode
  const activeEdge = focusedNode ? null : (focusedEdge ?? selectedEdge)
  const element = useMemo(() => selectedElement(rawGraph, activeNode, activeEdge), [rawGraph, activeNode, activeEdge])
  if (!element) return null

  const close = () => useKnowledgeGraphStore.getState().clearSelection()

  // Neighbour list for the node panel, matching graph_ui/src/components/graph/
  // PropertiesView.tsx's "Relations(within subgraph)" section: every edge
  // touching the selected node, resolved to the neighbour's display label.
  const relationships = element.type === 'node' && rawGraph
    ? rawGraph.edges
      .filter((edge) => edge.source === element.element.id || edge.target === element.element.id)
      .map((edge) => {
        const neighbourId = edge.source === element.element.id ? edge.target : edge.source
        const neighbour = rawGraph.nodes.find((node) => node.id === neighbourId)
        return { id: neighbourId, label: neighbour ? nodeDisplayName(neighbour) : neighbourId }
      })
    : []

  if (element.type === 'node') return <aside className="absolute top-3 right-3 z-20 w-80 max-h-[calc(100%-1.5rem)] overflow-auto rounded-lg border border-outline-variant/50 bg-surface-container-high/95 p-3 text-xs shadow-xl backdrop-blur">
    <SectionHeader title="Node" colorClass="text-sky-400" onClose={close}>
      <Action label="Expand" icon={<GitBranchPlus size={14} />} onClick={() => onExpand(element.element.id)} /><Action label="Prune" icon={<Scissors size={14} />} onClick={() => pruneNode(element.element.id)} />
    </SectionHeader>
    <PropertyRows properties={{ ID: element.element.id, Labels: element.element.labels.join(', '), Degree: element.element.degree }} />

    <h3 className="mb-1 mt-3 pl-1 text-sm font-bold tracking-wide text-amber-400">Properties</h3>
    <PropertyRows properties={element.element.properties} />

    {relationships.length > 0 && (
      <>
        <h3 className="mb-1 mt-3 pl-1 text-sm font-bold tracking-wide text-emerald-400">Relations (within subgraph)</h3>
        <div className="max-h-40 space-y-1 overflow-auto rounded bg-primary/5 p-1">
          {relationships.map(({ id, label }) => (
            <button
              key={id}
              type="button"
              onClick={() => useKnowledgeGraphStore.getState().selectNode(id, true)}
              className="flex w-full gap-2 rounded px-1 py-1 text-left hover:bg-surface-variant/50"
            >
              <span className="shrink-0 text-primary/80">Neighbour</span>
              <span className="min-w-0 flex-1 truncate text-on-surface-variant">{label}</span>
            </button>
          ))}
        </div>
      </>
    )}
  </aside>
  return <aside className="absolute top-3 right-3 z-20 w-80 max-h-[calc(100%-1.5rem)] overflow-auto rounded-lg border border-outline-variant/50 bg-surface-container-high/95 p-3 text-xs shadow-xl backdrop-blur">
    <SectionHeader title="Relationship" colorClass="text-violet-400" onClose={close} />
    <PropertyRows
      properties={{
        ID: element.element.id,
        ...(element.element.type ? { Type: element.element.type } : {}),
        Source: element.element.source,
        Target: element.element.target,
      }}
    />

    <h3 className="mb-1 mt-3 pl-1 text-sm font-bold tracking-wide text-amber-400">Properties</h3>
    <PropertyRows properties={element.element.properties} />
  </aside>
}

function SectionHeader({
  title, colorClass, onClose, children,
}: { title: string; colorClass: string; onClose: () => void; children?: ReactNode }) {
  return <div className="mb-2 flex items-center justify-between">
    <h3 className={`text-sm font-bold tracking-wide ${colorClass}`}>{title}</h3>
    <div className="flex items-center gap-2">
      {children}
      <button type="button" onClick={onClose} title="Close properties"><X size={15} /></button>
    </div>
  </div>
}
function Action({ label, icon, onClick }: { label: string; icon: ReactNode; onClick: () => void }) { return <button type="button" onClick={onClick} title={label} className="inline-flex size-7 items-center justify-center rounded border border-outline-variant/50 hover:bg-surface-variant">{icon}</button> }

function PropertyRows({ properties }: { properties: Record<string, unknown> }) {
  return <div className="max-h-96 space-y-1 overflow-auto rounded bg-primary/5 p-1">{Object.entries(properties).sort(([left], [right]) => left.localeCompare(right)).map(([name, value]) => <div key={name} className="flex gap-2 rounded px-1 py-1 hover:bg-surface-variant/50"><span className="shrink-0 text-primary/80">{name}</span><span className="min-w-0 flex-1 break-words text-on-surface-variant" title={formatValue(value)}>{formatValue(value)}</span></div>)}</div>
}

function formatValue(value: unknown): string {
  if (typeof value === 'string') return value.replaceAll('<SEP>', '; ')
  try { return JSON.stringify(value) } catch { return String(value) }
}
