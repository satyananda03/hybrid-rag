import { useMemo, useState } from 'react'
import { useKnowledgeGraphStore } from '@/features/knowledge_graph/store'

const MAX_RESULTS = 50

export function GraphSearch() {
  const rawGraph = useKnowledgeGraphStore((state) => state.rawGraph)
  const searchIndex = useKnowledgeGraphStore((state) => state.searchIndex)
  const selectNode = useKnowledgeGraphStore((state) => state.selectNode)
  const focusNode = useKnowledgeGraphStore((state) => state.focusNode)
  const [query, setQuery] = useState('')
  const [isOpen, setIsOpen] = useState(false)

  const results = useMemo(() => {
    if (!rawGraph) return []
    const normalized = query.trim()
    if (!normalized) return rawGraph.nodes.slice(0, MAX_RESULTS)

    const resultIds = searchIndex
      ? searchIndex.search(normalized).map((result) => String(result.id))
      : []
    const matchedIds = new Set(resultIds)
    const middleMatches = rawGraph.nodes
      .filter((node) => !matchedIds.has(node.id) && `${node.id} ${node.labels.join(' ')}`.toLowerCase().includes(normalized.toLowerCase()))
      .map((node) => node.id)

    return [...resultIds, ...middleMatches]
      .slice(0, MAX_RESULTS)
      .map((id) => rawGraph.nodes.find((node) => node.id === id))
      .filter((node): node is NonNullable<typeof node> => Boolean(node))
  }, [query, rawGraph, searchIndex])

  const selectResult = (nodeId: string) => {
    focusNode(null)
    selectNode(nodeId, true)
    setQuery('')
    setIsOpen(false)
  }

  return <div className="relative w-64">
    <input
      value={query}
      onChange={(event) => { setQuery(event.target.value); setIsOpen(true) }}
      onFocus={() => { focusNode(null); setIsOpen(true) }}
      onBlur={() => window.setTimeout(() => { focusNode(null); setIsOpen(false) }, 100)}
      onKeyDown={(event) => {
        if (event.key === 'Enter' && results[0]) { event.preventDefault(); selectResult(results[0].id) }
        if (event.key === 'Escape') { setIsOpen(false); focusNode(null) }
      }}
      className="w-full rounded border border-outline-variant/50 bg-surface-container-low px-2 py-1.5 text-sm text-on-surface outline-none focus:border-primary"
      placeholder="Find a node"
    />
    {isOpen && results.length > 0 && <div className="absolute z-30 mt-1 max-h-72 w-full overflow-y-auto rounded border border-outline-variant/50 bg-surface-container-high shadow-xl">{results.map((node) => <button key={node.id} type="button" onMouseDown={(event) => event.preventDefault()} onMouseEnter={() => focusNode(node.id)} onClick={() => selectResult(node.id)} className="block w-full truncate px-2 py-1.5 text-left text-xs text-on-surface hover:bg-surface-variant">{node.labels.join(', ') || node.id}</button>)}</div>}
  </div>
}
