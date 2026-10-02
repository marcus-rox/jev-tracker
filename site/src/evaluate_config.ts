import { REPO } from './types'

export type Source = 'kev' | 'laya' | 'api'
export const METHODS = ['noul_query_in_state', 'score_query_in_question'] as const
export type Method = (typeof METHODS)[number]
export const ISSUES_NEW = `${REPO}/issues/new?labels=evaluate`

// Same arms every committed config starts with; the new model is compared against them.
const BASELINE = `rerankers:
  prod:
    source: production
  jev_noul:
    source: answers
    method: noul_query_in_state
    path: data/jev/noul_query_in_state.json.gz
  jev_score:
    source: answers
    method: score_query_in_question
    path: data/jev/score_query_in_question.json.gz`

const ARM: Record<Source, (model: string, keyEnv: string) => string> = {
  kev: (model) => `    source: kev
    model: ${model}
    max_items: 12
    max_chars: 12000
    shards: 3
    concurrency: 16`,
  laya: (model) => `    source: laya
    model: ${model}
    shards: 3
    forward_batch: 64`,
  api: (model, keyEnv) => `    source: api
    url: ${model}
    model: ${keyEnv ? keyEnv.toLowerCase().replace(/_api_key$/, '') : 'default'}
    secret: ${keyEnv ? keyEnv.toLowerCase().replace(/_/g, '-') : 'api-key'}
    key_env: ${keyEnv || 'API_KEY'}
    max_items: 25
    max_chars: 24000
    concurrency: 8`,
}

export function configYaml(name: string, source: Source, model: string, methods: Method[], keyEnv: string): string {
  const arms = methods.map((m) => `  ${name}_${m.split('_')[0]}:\n${ARM[source](model, keyEnv)}\n    method: ${m}`).join('\n')
  return `name: ${name}\ncases: null\nk: {start: 50, stop: 200, step: 10}\n${BASELINE}\n${arms}\n`
}

export function issueUrl(name: string, yaml: string): string {
  const body = `Please evaluate this model on the 75 frozen cases.\n\n\`\`\`yaml\n${yaml}\`\`\`\n`
  return `${ISSUES_NEW}&title=${encodeURIComponent(`evaluate: ${name}`)}&body=${encodeURIComponent(body)}`
}
