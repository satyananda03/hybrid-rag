import { useEffect, useState } from 'react'
import { getPopularGraphLabels, searchGraphLabels } from '@/features/knowledge_graph/api'
import { useKnowledgeGraphStore } from '@/features/knowledge_graph/store'
import type { GraphWorkspace } from '@/features/knowledge_graph/types'

export function GraphLabelSearch({ workspace }: { workspace: GraphWorkspace }) {
  const label = useKnowledgeGraphStore((state) => state.settings.label)
  const updateSettings = useKnowledgeGraphStore((state) => state.updateSettings)
  const [value, setValue] = useState(label)
  const [suggestions, setSuggestions] = useState<string[]>([])

  useEffect(() => {
    let active = true
    void getPopularGraphLabels(workspace)
      .then((labels) => { if (active) setSuggestions(labels) })
      .catch(() => { if (active) setSuggestions([]) })
    return () => { active = false }
  }, [workspace])

  const [previousLabel, setPreviousLabel] = useState(label)
  if (label !== previousLabel) {
    setPreviousLabel(label)
    setValue(label)
  }

  const applyLabel = (nextValue: string) => {
    const nextLabel = nextValue.trim() || '*'
    if (nextLabel !== label) updateSettings({ label: nextLabel })
  }

  async function search(valueToSearch: string) {
    setValue(valueToSearch)
    if (!valueToSearch.trim()) return
    try { setSuggestions(await searchGraphLabels(workspace, valueToSearch)) } catch { setSuggestions([]) }
  }

  return <label className="flex min-w-52 flex-col gap-1 text-xs text-on-surface-variant">
    <input
      list="knowledge-graph-labels"
      value={value}
      onChange={(event) => {
        const nextValue = event.target.value
        void search(nextValue)
        if (nextValue === '*' || suggestions.includes(nextValue)) applyLabel(nextValue)
      }}
      onKeyDown={(event) => { if (event.key === 'Enter') applyLabel(value) }}
      onBlur={() => applyLabel(value)}
      className="rounded border border-outline-variant/50 bg-surface-container-low px-2 py-1.5 text-sm text-on-surface outline-none focus:border-primary"
      placeholder="Search graph labels"
    />
    <datalist id="knowledge-graph-labels">{['*', ...suggestions].map((item) => <option key={item} value={item} />)}</datalist>
  </label>
}
