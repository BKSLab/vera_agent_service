# Контракт сайта с Feedback API Agent Service

Документ описывает два запроса, которые серверная часть сайта отправляет
в Agent Service для сохранения обратной связи.

## 1. Подключение

Базовый адрес Agent Service:

```text
http://91.218.115.104:8010
```

Все запросы отправляются с заголовками:

```http
Content-Type: application/json
X-API-Key: <API_KEY Agent Service>
```

Значение `X-API-Key` должно храниться только в переменных окружения
серверной части сайта. Отправлять этот ключ в браузер нельзя.

## 2. Идентификаторы

Сайт использует те же идентификаторы, с которыми работает чат:

- `session_id` — идентификатор текущей сессии диалога;
- `request_id` — идентификатор конкретного запроса пользователя и ответа
  Веры;
- `submission_id` — уникальный идентификатор одной отправки развёрнутой
  анкеты.

`session_id` и `request_id` нельзя генерировать заново при отправке оценки:
они должны совпадать с идентификаторами уже обработанного сообщения.

`submission_id` генерируется сайтом перед первой отправкой анкеты. При
повторе того же запроса после timeout или сетевой ошибки сайт обязан
использовать прежний `submission_id`.

## 3. Оценка конкретного ответа

Создаёт оценку ответа или изменяет уже существующую оценку, например
`up` на `down`.

```http
PUT http://91.218.115.104:8010/api/v1/feedback/message
```

Тело запроса:

```json
{
  "session_id": "conversation-uuid",
  "request_id": "message-uuid",
  "value": "down"
}
```

Поля:

| Поле | Тип | Обязательно | Ограничения |
|---|---|---:|---|
| `session_id` | string | да | от 1 до 100 символов |
| `request_id` | string | да | от 1 до 100 символов |
| `value` | string | да | только `up` или `down` |

Успешный ответ — `200 OK`:

```json
{
  "id": "f6fdde40-982c-43cb-b15f-f5851ab22035",
  "session_id": "conversation-uuid",
  "request_id": "message-uuid",
  "value": "down",
  "review_status": "new",
  "created_at": "2026-07-29T12:00:00Z",
  "updated_at": "2026-07-29T12:00:00Z"
}
```

Правила:

- оценивать можно только уже завершённый ответ;
- `request_id` должен относиться к переданному `session_id`;
- повторный запрос для того же ответа не создаёт вторую запись;
- повторный запрос может изменить значение оценки;
- лимит Agent Service — 60 запросов в минуту.

Пример вызова из серверного кода:

```typescript
const response = await fetch(
  "http://91.218.115.104:8010/api/v1/feedback/message",
  {
    method: "PUT",
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": process.env.VERA_AGENT_API_KEY!,
    },
    body: JSON.stringify({
      session_id: sessionId,
      request_id: requestId,
      value: "down",
    }),
  },
);
```

## 4. Развёрнутый отзыв по сессии

Сохраняет анкету по существующей сессии.

```http
POST http://91.218.115.104:8010/api/v1/feedback/session
```

Тело запроса:

```json
{
  "session_id": "conversation-uuid",
  "submission_id": "feedback-submission-uuid",
  "audience": "employer",
  "usefulness": 3,
  "trust": 2,
  "comment": "Не хватило пояснения по источнику",
  "contact_email": "user@example.ru"
}
```

Поля:

| Поле | Тип | Обязательно | Ограничения |
|---|---|---:|---|
| `session_id` | string | да | от 1 до 100 символов |
| `submission_id` | string | да | от 1 до 100 символов |
| `audience` | string или `null` | нет | `seeker`, `employer` или `other` |
| `usefulness` | integer или `null` | нет | от 1 до 5 |
| `trust` | integer или `null` | нет | от 1 до 5 |
| `comment` | string или `null` | нет | не более 4000 символов |
| `contact_email` | string или `null` | нет | валидный email, не более 320 символов |

Необязательные поля можно не передавать либо передавать как `null`.

Успешный ответ — `201 Created`:

```json
{
  "id": "ef5dcc40-ed5b-4842-8df4-d52d4378a8d0",
  "session_id": "conversation-uuid",
  "submission_id": "feedback-submission-uuid",
  "review_status": "new",
  "created_at": "2026-07-29T12:05:00Z"
}
```

Правила:

- `session_id` должен существовать в Agent Service;
- один `submission_id` создаёт только один отзыв;
- повтор с теми же `submission_id` и `session_id` возвращает ранее
  сохранённый отзыв;
- повтор не изменяет содержимое ранее сохранённого отзыва;
- использование одного `submission_id` для другой сессии возвращает
  ошибку `409`;
- лимит Agent Service — 10 запросов в минуту;
- Agent Service сохраняет отзыв в БД, но не отправляет его по email.

Пример вызова из серверного кода:

```typescript
const response = await fetch(
  "http://91.218.115.104:8010/api/v1/feedback/session",
  {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": process.env.VERA_AGENT_API_KEY!,
    },
    body: JSON.stringify({
      session_id: sessionId,
      submission_id: submissionId,
      audience: "employer",
      usefulness: 3,
      trust: 2,
      comment: "Не хватило пояснения по источнику",
      contact_email: "user@example.ru",
    }),
  },
);
```

## 5. Ошибки

Обычный ответ с ошибкой:

```json
{
  "detail": "Описание ошибки."
}
```

| HTTP-код | Значение |
|---:|---|
| `401` | Передан неверный `X-API-Key` |
| `404` | Сессия или оцениваемый ответ не найдены |
| `409` | Ответ относится к другой сессии, ответ ещё не завершён либо `submission_id` относится к другой сессии |
| `422` | Не передан обязательный заголовок/поле или тело не прошло валидацию |
| `429` | Превышен лимит запросов |
| `500` | Внутренняя ошибка сохранения |

При `422` поле `detail` содержит стандартный список ошибок FastAPI.

## 6. Поведение сайта при ошибках

- `200` и `201` считать успешным сохранением.
- При сетевой ошибке или `500` разрешено повторить запрос с теми же
  идентификаторами.
- Для повторной отправки анкеты нельзя создавать новый `submission_id`.
- `401`, `404`, `409` и `422` автоматически не повторять.
- Текст вопроса и ответа не передавать: Agent Service уже сохраняет их
  при обработке сообщения.
