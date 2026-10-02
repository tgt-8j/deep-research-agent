<script setup lang="ts">
import { nextTick, ref, computed, onMounted } from 'vue'

// ─── 认证状态 ────────────────────────────────────────────────
const TOKEN_KEY = 'deep_research_jwt_token'
const USER_KEY = 'deep_research_user_info'

interface UserInfo {
  user_id: string
  tenant_id: string
  name: string
}

const token = ref<string>(localStorage.getItem(TOKEN_KEY) || '')
const isLoggedIn = computed(() => !!token.value)

const userInfo = ref<UserInfo | null>(
  (() => {
    try { return JSON.parse(localStorage.getItem(USER_KEY) || 'null') }
    catch { return null }
  })()
)

const showLogin = ref(!isLoggedIn.value)

const loginForm = ref({ api_key: '', tenant_id: 'tenant_demo' })
const loginError = ref('')
const loginLoading = ref(false)

async function handleLogin() {
  loginError.value = ''
  loginLoading.value = true
  try {
    const res = await fetch('http://127.0.0.1:8000/api/v1/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        api_key: loginForm.value.api_key,
        tenant_id: loginForm.value.tenant_id,
      }),
    })
    if (!res.ok) {
      const data = await res.json().catch(() => ({}))
      throw new Error(data.detail || `HTTP ${res.status}`)
    }
    const data = await res.json()
    token.value = data.access_token
    localStorage.setItem(TOKEN_KEY, data.access_token)
    userInfo.value = {
      user_id: data.user_id,
      tenant_id: data.tenant_id,
      name: data.user_id,
    }
    localStorage.setItem(USER_KEY, JSON.stringify(userInfo.value))
    showLogin.value = false
  } catch (e: any) {
    loginError.value = e.message || '登录失败'
  } finally {
    loginLoading.value = false
  }
}

function handleLogout() {
  token.value = ''
  userInfo.value = null
  localStorage.removeItem(TOKEN_KEY)
  localStorage.removeItem(USER_KEY)
  showLogin.value = true
  messages.value = []
}

function getValidToken(): string | null {
  return token.value || localStorage.getItem(TOKEN_KEY)
}

// ─── 聊天状态 ────────────────────────────────────────────────
type StreamEvent = {
  type: 'status' | 'phase' | 'route' | 'final' | 'error'
  message?: string
  final?: string
  node?: string
}

type ChatMessage = {
  id: string
  role: 'user' | 'assistant' | 'status'
  content: string
}

const userId = computed(() => userInfo.value?.user_id || 'user01')
const tenantId = computed(() => userInfo.value?.tenant_id || 'default_tenant')
const threadId = ref('thread01')
const query = ref('')
const loading = ref(false)
const errorMessage = ref('')
const messageListRef = ref<HTMLElement | null>(null)
const composerRef = ref<HTMLTextAreaElement | null>(null)
const progressLogs = ref<string[]>([])

const starterPrompts = [
  { title: '深度调研', prompt: '请调研"企业知识库 Agent 平台"市场，按市场规模、主要竞品、收费模式三部分输出，并在每部分附上可追溯来源链接。' },
  { title: '方案对比', prompt: '我们要做多 Agent 研究助手，请对比"纯大模型直答""RAG 单 Agent""多 Agent 协作"三种方案，给出优缺点、适用场景与推荐结论。' },
  { title: '知识问答', prompt: '请解释这个项目里"意图分流"的作用，以及简单问题和复杂问题分别会走哪条链路。' },
  { title: '落地计划', prompt: '请把"上线一个可用的 DeepResearch MVP"拆成两周计划，按每天输出任务、验收标准和风险点。' },
]

const capabilityHighlights = [
  { title: '多智能体编排', desc: '自动完成规划、检索、证据裁判、分析与写作，减少手工研究路径。' },
  { title: '双源检索融合', desc: '网络信息与本地知识库并行召回，输出结论同时保留来源可追溯性。' },
  { title: '会话记忆增强', desc: '跨轮次继承用户偏好与历史任务，持续提升回答一致性和效率。' },
]

const landingMetrics = [
  { label: '执行模式', value: 'Quick + Deep' },
  { label: '检索来源', value: 'Web + Local' },
  { label: '输出风格', value: '结论 + 证据' },
]

const messages = ref<ChatMessage[]>([
  { id: `m-${Date.now()}`, role: 'assistant', content: '你好，我是 DeepResearch。你可以直接提问，我会根据意图自动走快速回答或完整研究链路。' },
])

// ─── 工具函数 ────────────────────────────────────────────────
const escapeHtml = (value: string): string =>
  value.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;').replaceAll("'", '&#39;')

const markdownToHtml = (markdown: string): string => {
  const codeBlocks: string[] = []
  let text = markdown.replace(/```([\s\S]*?)```/g, (_, block) => {
    const index = codeBlocks.length
    codeBlocks.push(`<pre><code>${escapeHtml(String(block).trim())}</code></pre>`)
    return `@@CODE_BLOCK_${index}@@`
  })
  const lines = text.split('\n')
  const out: string[] = []
  let inList = false
  const closeList = () => { if (inList) { out.push('</ul>'); inList = false } }
  for (const rawLine of lines) {
    const line = rawLine.trim()
    if (!line) { closeList(); continue }
    if (line.startsWith('# ')) { closeList(); out.push(`<h1>${escapeHtml(line.slice(2))}</h1>`); continue }
    if (line.startsWith('## ')) { closeList(); out.push(`<h2>${escapeHtml(line.slice(3))}</h2>`); continue }
    if (line.startsWith('### ')) { closeList(); out.push(`<h3>${escapeHtml(line.slice(4))}</h3>`); continue }
    if (line.startsWith('- ') || line.startsWith('* ')) {
      if (!inList) { out.push('<ul>'); inList = true }
      out.push(`<li>${escapeHtml(line.slice(2))}</li>`); continue
    }
    closeList()
    out.push(`<p>${escapeHtml(line)}</p>`)
  }
  closeList()
  let html = out.join('')
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>')
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>')
  html = html.replace(/\[([^[\]]+)\]\((https?:\/\/[^)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>')
  html = html.replace(/@@CODE_BLOCK_(\d+)@@/g, (_, idx) => codeBlocks[Number(idx)] || '')
  return html
}

const renderMessageHtml = (message: ChatMessage) => markdownToHtml(message.content || '')

const scrollToBottom = async () => {
  await nextTick()
  const el = messageListRef.value
  if (el) el.scrollTop = el.scrollHeight
}

const createNewChat = () => {
  messages.value = [{ id: `m-${Date.now()}`, role: 'assistant', content: '已开始新会话。你可以继续提问。' }]
  progressLogs.value = []
  errorMessage.value = ''
  query.value = ''
}

const usePrompt = async (prompt: string) => {
  query.value = prompt
  errorMessage.value = ''
  await nextTick()
  composerRef.value?.focus()
}

const applyStarterByIndex = (index: number) => {
  const target = starterPrompts[index]
  if (!target) return
  usePrompt(target.prompt)
}

const pushProgress = (message: string) => {
  const msg = message.trim()
  if (!msg) return
  const last = progressLogs.value[progressLogs.value.length - 1]
  if (last === msg) return
  progressLogs.value.push(msg)
  if (progressLogs.value.length > 6) progressLogs.value = progressLogs.value.slice(-6)
}

const runResearch = async () => {
  const userText = query.value.trim()
  if (!userText || loading.value) return
  const currentToken = getValidToken()
  if (!currentToken) {
    showLogin.value = true
    return
  }
  loading.value = true
  errorMessage.value = ''
  progressLogs.value = []
  query.value = ''
  messages.value.push({ id: `u-${Date.now()}`, role: 'user', content: userText })
  const statusId = `s-${Date.now()}`
  messages.value.push({ id: statusId, role: 'status', content: '正在初始化执行链路...' })
  const renderStatusText = () => {
    const statusMessage = messages.value.find((item) => item.id === statusId)
    if (!statusMessage) return
    const latest = progressLogs.value.slice(-8)
    statusMessage.content = ['正在处理中...', ...latest].map((line) => `- ${line}`).join('\n')
  }
  renderStatusText()
  await scrollToBottom()
  try {
    const response = await fetch('/api/v1/research/stream', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${currentToken}`,
      },
      body: JSON.stringify({
        query: userText,
        user_id: userId.value,
        thread_id: threadId.value,
        tenant_id: tenantId.value,
      }),
    })
    if (!response.ok) {
      if (response.status === 401) {
        showLogin.value = true
        throw new Error('认证已过期，请重新登录')
      }
      const text = await response.text()
      throw new Error(text || `请求失败: ${response.status}`)
    }
    if (!response.body) throw new Error('流式响应不可用')
    const reader = response.body.getReader()
    const decoder = new TextDecoder('utf-8')
    let buffer = ''
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const parts = buffer.split('\n\n')
      buffer = parts.pop() || ''
      for (const part of parts) {
        if (!part.startsWith('data: ')) continue
        const jsonText = part.slice(6).trim()
        if (!jsonText) continue
        const event = JSON.parse(jsonText) as StreamEvent
        if (event.type === 'status' || event.type === 'phase' || event.type === 'route') {
          const prefix = event.type === 'phase' && event.node ? `[${event.node}] ` : ''
          pushProgress(`${prefix}${event.message || ''}`)
          renderStatusText()
        }
        if (event.type === 'final') {
          messages.value = messages.value.filter((item) => item.id !== statusId)
          messages.value.push({ id: `a-${Date.now()}`, role: 'assistant', content: event.final || '已完成，但未返回正文。' })
        }
        if (event.type === 'error') throw new Error(event.message || '服务端执行异常')
      }
      await scrollToBottom()
    }
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '请求失败'
    messages.value = messages.value.filter((item) => item.id !== statusId)
    messages.value.push({ id: `e-${Date.now()}`, role: 'assistant', content: `请求失败：${errorMessage.value}` })
  } finally {
    loading.value = false
    await scrollToBottom()
  }
}
</script>

<template>
  <!-- ══════════════ 登录页面 ══════════════ -->
  <div v-if="showLogin" class="login-shell">
    <div class="login-card">
      <div class="login-header">
        <div class="login-logo">DR</div>
        <h1>DeepResearch</h1>
        <p>企业级多智能体研究工作台</p>
      </div>
      <div class="login-form">
        <div class="form-group">
          <label>API Key</label>
          <input v-model="loginForm.api_key" type="password" placeholder="输入 API Key" @keydown.enter="handleLogin" />
        </div>
        <div class="form-group">
          <label>租户 ID</label>
          <select v-model="loginForm.tenant_id">
            <option value="tenant_demo">tenant_demo（演示）</option>
          </select>
        </div>
        <p v-if="loginError" class="login-error">{{ loginError }}</p>
        <button class="login-btn" :disabled="loginLoading" @click="handleLogin">
          {{ loginLoading ? '登录中...' : '登录' }}
        </button>
        <p class="login-hint">演示账号：API Key = demo_key，租户 = tenant_demo</p>
      </div>
    </div>
  </div>

  <!-- ══════════════ 主界面 ══════════════ -->
  <div v-else class="chat-shell">
    <aside class="chat-sidebar">
      <div class="sidebar-brand">
        <p class="brand-badge">AI Copilot</p>
        <h1>DeepResearch</h1>
        <p class="brand-desc">多智能体研究工作台，支持快速回答与深度调研。</p>
      </div>
      <div class="sidebar-head">
        <button class="new-chat-btn" @click="createNewChat">新建会话</button>
      </div>
      <div class="quick-entry">
        <p class="section-title">推荐起手问题</p>
        <button v-for="item in starterPrompts.slice(0, 3)" :key="item.title" class="quick-entry-btn" @click="usePrompt(item.prompt)">
          {{ item.title }}
        </button>
      </div>
      <div class="settings-group">
        <label>User ID</label>
        <input :value="userId" class="sidebar-input" readonly />
      </div>
      <div class="settings-group">
        <label>Thread ID</label>
        <input v-model="threadId" class="sidebar-input" />
      </div>
      <div class="settings-group">
        <label>Tenant ID</label>
        <input :value="tenantId" class="sidebar-input" readonly />
      </div>
      <div class="user-info-bar">
        <span class="user-avatar">{{ (userInfo?.name || 'U')[0].toUpperCase() }}</span>
        <span class="user-name">{{ userInfo?.name || 'Unknown' }}</span>
        <button class="logout-btn" @click="handleLogout">退出</button>
      </div>
      <p class="hint-text">已认证 · JWT Token 已注入请求头</p>
    </aside>

    <main class="chat-main">
      <header class="main-header">
        <div>
          <h2>DeepResearch Enterprise Workspace</h2>
          <p>面向业务团队的企业级智能研究台，支持从问题定义到结论落地的完整链路。</p>
        </div>
        <div class="header-tags">
          <span>Evidence-Driven</span>
          <span>Structured Output</span>
          <span>Memory-Powered</span>
        </div>
      </header>
      <div ref="messageListRef" class="message-list">
        <section v-if="messages.length <= 1" class="onboarding-panel">
          <div class="hero-panel">
            <p class="hero-badge">商业研究 · 策略分析 · 知识问答</p>
            <h3>第一步先讲清目标，再交给 DeepResearch 自动推进</h3>
            <p class="hero-desc">推荐提问结构：目标 + 背景约束 + 期望输出。系统会自动选择快速回答或深度研究链路。</p>
            <div class="hero-actions">
              <button class="hero-btn primary" @click="applyStarterByIndex(0)">快速开始调研</button>
              <button class="hero-btn" @click="applyStarterByIndex(1)">查看方案对比</button>
            </div>
            <div class="metric-grid">
              <article v-for="item in landingMetrics" :key="item.label">
                <p>{{ item.label }}</p>
                <strong>{{ item.value }}</strong>
              </article>
            </div>
          </div>
          <div class="capability-grid">
            <article v-for="item in capabilityHighlights" :key="item.title" class="capability-card">
              <h4>{{ item.title }}</h4>
              <p>{{ item.desc }}</p>
            </article>
          </div>
          <div class="guide-panel">
            <h4>提问指南</h4>
            <div class="guide-grid">
              <article><h5>1. 说明目标</h5><p>你要解决什么问题、面向谁、希望达到什么结果。</p></article>
              <article><h5>2. 提供上下文</h5><p>给出已知信息、时间范围、数据口径、业务限制。</p></article>
              <article><h5>3. 指定输出</h5><p>例如"表格输出""附来源链接""分点行动清单"。</p></article>
            </div>
          </div>
          <div class="prompt-list">
            <button v-for="item in starterPrompts" :key="item.prompt" class="prompt-chip" @click="usePrompt(item.prompt)">
              {{ item.prompt }}
            </button>
          </div>
        </section>
        <div v-for="message in messages" :key="message.id" class="message-row" :class="`role-${message.role}`">
          <div class="avatar">{{ message.role === 'user' ? '你' : message.role === 'status' ? '...' : 'AI' }}</div>
          <div class="bubble markdown-body" v-html="renderMessageHtml(message)"></div>
        </div>
      </div>
      <div class="composer">
        <textarea
          v-model="query"
          ref="composerRef"
          class="composer-input"
          :disabled="loading"
          placeholder="输入你的问题，回车发送（Shift + Enter 换行）"
          @keydown.enter.exact.prevent="runResearch"
        />
        <button class="send-btn" :disabled="loading || !query.trim()" @click="runResearch">
          {{ loading ? '处理中...' : '发送' }}
        </button>
      </div>
      <p v-if="errorMessage" class="error">{{ errorMessage }}</p>
    </main>
  </div>
</template>

<style scoped>
/* ════════ 登录页 ════════ */
.login-shell {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background: radial-gradient(ellipse at 30% 20%, #eef4ff 0%, #f0f4ff 40%, #f8f9fc 100%);
  padding: 20px;
}
.login-card {
  width: 100%;
  max-width: 400px;
  background: #fff;
  border-radius: 20px;
  box-shadow: 0 20px 60px rgba(15, 35, 95, 0.12);
  padding: 40px 32px;
}
.login-header { text-align: center; margin-bottom: 32px; }
.login-logo {
  width: 52px; height: 52px; border-radius: 14px;
  background: linear-gradient(135deg, #3b82f6, #1d4ed8);
  display: inline-flex; align-items: center; justify-content: center;
  color: #fff; font-weight: 800; font-size: 18px; margin-bottom: 12px;
}
.login-header h1 { margin: 0 0 4px; font-size: 24px; color: #0f172a; }
.login-header p { margin: 0; color: #64748b; font-size: 14px; }
.form-group { margin-bottom: 16px; }
.form-group label { display: block; font-size: 13px; font-weight: 600; color: #334155; margin-bottom: 6px; }
.form-group input, .form-group select {
  width: 100%; padding: 10px 14px; border: 1.5px solid #e2e8f0; border-radius: 10px;
  font-size: 14px; outline: none; transition: border-color .2s;
}
.form-group input:focus, .form-group select:focus { border-color: #3b82f6; }
.login-error { color: #dc2626; font-size: 13px; margin: 8px 0; }
.login-btn {
  width: 100%; padding: 12px; border: none; border-radius: 10px;
  background: linear-gradient(135deg, #3b82f6, #1d4ed8);
  color: #fff; font-size: 15px; font-weight: 600; cursor: pointer; transition: opacity .2s;
}
.login-btn:disabled { opacity: 0.6; cursor: not-allowed; }
.login-btn:hover:not(:disabled) { opacity: 0.9; }
.login-hint { text-align: center; font-size: 12px; color: #94a3b8; margin-top: 16px; }

/* ════════ 主界面（保持原有样式）════════ */
.chat-shell { height: 100vh; display: flex; background: radial-gradient(ellipse at 30% 20%, #eef4ff 0%, #f0f4ff 40%, #f8f9fc 100%); overflow: hidden; }
.chat-sidebar { width: 260px; background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%); display: flex; flex-direction: column; padding: 16px; gap: 12px; overflow-y: auto; }
.sidebar-brand { padding: 8px 0 12px; border-bottom: 1px solid rgba(255,255,255,0.08); }
.brand-badge { font-size: 11px; color: #60a5fa; font-weight: 700; letter-spacing: 0.5px; margin: 0; }
.sidebar-brand h1 { font-size: 18px; color: #f8fafc; margin: 4px 0; }
.brand-desc { font-size: 12px; color: #94a3b8; margin: 4px 0 0; line-height: 1.5; }
.sidebar-head { padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.08); }
.new-chat-btn { width: 100%; padding: 8px; border: 1.5px solid rgba(96,165,250,0.4); border-radius: 8px; background: transparent; color: #93c5fd; font-size: 13px; cursor: pointer; transition: all .2s; }
.new-chat-btn:hover { background: rgba(59,130,246,0.15); border-color: #3b82f6; }
.quick-entry { padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.08); }
.section-title { font-size: 11px; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin: 0 0 8px; }
.quick-entry-btn { width: 100%; padding: 8px 10px; border: none; border-radius: 6px; background: rgba(255,255,255,0.05); color: #cbd5e1; font-size: 12px; text-align: left; cursor: pointer; margin-bottom: 4px; transition: all .2s; }
.quick-entry-btn:hover { background: rgba(59,130,246,0.2); color: #93c5fd; }
.settings-group { padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.08); }
.settings-group label { display: block; font-size: 11px; color: #64748b; font-weight: 600; margin-bottom: 4px; }
.sidebar-input { width: 100%; padding: 6px 10px; border: 1px solid rgba(255,255,255,0.1); border-radius: 6px; background: rgba(255,255,255,0.05); color: #e2e8f0; font-size: 12px; outline: none; }
.sidebar-input:read-only { background: rgba(255,255,255,0.03); color: #94a3b8; }
.user-info-bar { display: flex; align-items: center; gap: 8px; padding: 10px; background: rgba(59,130,246,0.1); border-radius: 8px; border: 1px solid rgba(59,130,246,0.2); }
.user-avatar { width: 28px; height: 28px; border-radius: 50%; background: linear-gradient(135deg, #3b82f6, #1d4ed8); display: flex; align-items: center; justify-content: center; color: #fff; font-size: 12px; font-weight: 700; flex-shrink: 0; }
.user-name { flex: 1; color: #e2e8f0; font-size: 13px; font-weight: 500; }
.logout-btn { padding: 4px 10px; border: 1px solid rgba(239,68,68,0.4); border-radius: 6px; background: transparent; color: #f87171; font-size: 11px; cursor: pointer; transition: all .2s; }
.logout-btn:hover { background: rgba(239,68,68,0.15); }
.hint-text { font-size: 11px; color: #475569; text-align: center; padding: 8px; background: rgba(59,130,246,0.06); border-radius: 6px; }

.chat-main { flex: 1; display: flex; flex-direction: column; background: #f8f9fc; overflow: hidden; }
.main-header { padding: 20px 28px 16px; background: #fff; border-bottom: 1px solid #e2e8f0; }
.main-header h2 { margin: 0 0 4px; font-size: 18px; color: #0f172a; }
.main-header p { margin: 0 0 12px; font-size: 13px; color: #64748b; }
.header-tags { display: flex; gap: 8px; }
.header-tags span { padding: 3px 10px; border-radius: 20px; background: #eff6ff; color: #3b82f6; font-size: 11px; font-weight: 600; }
.message-list { flex: 1; overflow-y: auto; padding: 24px 28px; }
.onboarding-panel { max-width: 760px; margin: 0 auto; }
.hero-panel { text-align: center; padding: 32px 20px; background: #fff; border-radius: 16px; border: 1px solid #e2e8f0; margin-bottom: 20px; }
.hero-badge { font-size: 12px; color: #3b82f6; font-weight: 700; letter-spacing: 0.5px; margin: 0 0 12px; }
.hero-panel h3 { margin: 0 0 8px; font-size: 22px; color: #0f172a; }
.hero-desc { font-size: 14px; color: #64748b; margin: 0 0 20px; line-height: 1.6; }
.hero-actions { display: flex; gap: 12px; justify-content: center; margin-bottom: 24px; }
.hero-btn { padding: 10px 20px; border-radius: 10px; font-size: 14px; font-weight: 600; cursor: pointer; transition: all .2s; border: 1.5px solid #e2e8f0; background: #fff; color: #334155; }
.hero-btn.primary { background: linear-gradient(135deg, #3b82f6, #1d4ed8); color: #fff; border-color: transparent; }
.hero-btn:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(59,130,246,0.2); }
.metric-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
.metric-grid article { padding: 12px; background: #f8fafc; border-radius: 8px; text-align: center; }
.metric-grid p { margin: 0 0 4px; font-size: 12px; color: #64748b; }
.metric-grid strong { font-size: 14px; color: #0f172a; }
.capability-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 20px; }
.capability-card { padding: 16px; background: #fff; border-radius: 12px; border: 1px solid #e2e8f0; }
.capability-card h4 { margin: 0 0 6px; font-size: 14px; color: #0f172a; }
.capability-card p { margin: 0; font-size: 12px; color: #64748b; line-height: 1.5; }
.guide-panel { background: #fff; border-radius: 12px; border: 1px solid #e2e8f0; padding: 20px; margin-bottom: 20px; }
.guide-panel h4 { margin: 0 0 16px; font-size: 15px; color: #0f172a; }
.guide-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
.guide-grid article h5 { margin: 0 0 4px; font-size: 13px; color: #3b82f6; }
.guide-grid article p { margin: 0; font-size: 12px; color: #64748b; line-height: 1.5; }
.prompt-list { display: flex; flex-direction: column; gap: 8px; }
.prompt-chip { padding: 12px 16px; background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; text-align: left; font-size: 13px; color: #334155; cursor: pointer; transition: all .2s; }
.prompt-chip:hover { border-color: #3b82f6; background: #eff6ff; color: #1d4ed8; }
.message-row { display: flex; gap: 12px; margin-bottom: 16px; max-width: 80%; }
.message-row.role-user { margin-left: auto; flex-direction: row-reverse; }
.message-row.role-status { max-width: 90%; }
.avatar { width: 32px; height: 32px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 700; flex-shrink: 0; }
.role-user .avatar { background: linear-gradient(135deg, #3b82f6, #1d4ed8); color: #fff; }
.role-assistant .avatar { background: linear-gradient(135deg, #0ea5e9, #22c55e); color: #fff; }
.role-status .avatar { background: #f1f5f9; color: #94a3b8; }
.bubble { padding: 12px 16px; border-radius: 14px; line-height: 1.6; font-size: 14px; }
.role-user .bubble { background: linear-gradient(135deg, #3b82f6, #1d4ed8); color: #fff; border-top-right-radius: 4px; }
.role-assistant .bubble { background: #fff; color: #1e293b; border: 1px solid #e2e8f0; border-top-left-radius: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.04); }
.role-status .bubble { background: #f8fafc; color: #64748b; border: 1px solid #e2e8f0; font-size: 12px; white-space: pre-wrap; }
.bubble :deep(p) { margin: 0 0 8px; }
.bubble :deep(p:last-child) { margin: 0; }
.bubble :deep(pre) { background: #f4f4f5; padding: 10px; border-radius: 6px; overflow-x: auto; font-size: 12px; }
.bubble :deep(code) { font-family: monospace; }
.bubble :deep(a) { color: #3b82f6; }
.composer { padding: 16px 28px 20px; background: #fff; border-top: 1px solid #e2e8f0; display: flex; gap: 12px; align-items: flex-end; }
.composer-input { flex: 1; padding: 12px 16px; border: 1.5px solid #e2e8f0; border-radius: 12px; font-size: 14px; resize: none; outline: none; min-height: 48px; max-height: 120px; line-height: 1.5; transition: border-color .2s; }
.composer-input:focus { border-color: #3b82f6; }
.composer-input:disabled { background: #f8fafc; }
.send-btn { padding: 12px 24px; border: none; border-radius: 10px; background: linear-gradient(135deg, #3b82f6, #1d4ed8); color: #fff; font-size: 14px; font-weight: 600; cursor: pointer; transition: opacity .2s; white-space: nowrap; }
.send-btn:disabled { opacity: 0.5; cursor: not-allowed; }
.send-btn:hover:not(:disabled) { opacity: 0.9; }
.error { margin: 0 28px 16px; padding: 10px 16px; background: #fef2f2; border: 1px solid #fecaca; border-radius: 8px; color: #dc2626; font-size: 13px; }

@media (max-width: 768px) {
  .chat-shell { flex-direction: column; }
  .chat-sidebar { width: 100%; max-height: 200px; }
  .metric-grid, .capability-grid, .guide-grid { grid-template-columns: 1fr; }
}
</style>
