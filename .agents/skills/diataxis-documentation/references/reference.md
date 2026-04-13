# Writing Reference Documentation

## What is Reference Documentation?

Reference documentation is **information-oriented** documentation that provides accurate, complete, and reliable technical descriptions of the machinery.

**Key characteristics:**
- Information-oriented
- Comprehensive and accurate
- Structured like the product
- Neutral and objective
- Factual descriptions

**User context:**
- Has working knowledge
- Working mode (at work, not learning)
- Needs to look up specific information
- Needs facts quickly

**User question:** "What exactly is...?" or "What are the parameters for...?"

## Purpose and Goals

### Primary Purpose
Provide accurate, complete technical information that users can quickly look up while working.

### Success Criteria
- User finds needed information quickly
- Information is accurate and complete
- User can trust the documentation
- Structure matches user's mental model
- Information is easy to scan and search

### What Reference Documentation Is NOT
- Not tutorials (those teach through doing)
- Not how-to guides (those solve specific problems)
- Not explanatory (those provide understanding)
- Not opinionated (reference is neutral)
- Not selective (reference is comprehensive)

## Core Principles

### 1. Describe, Don't Instruct

Reference describes what something is or does, not how to use it.

**Good:**
```markdown
## authenticate(credentials)

Validates user credentials and returns an authentication token.

**Parameters:**
- `credentials` (Object): User login credentials
  - `username` (string): User's username
  - `password` (string): User's password

**Returns:** (string) Authentication token

**Throws:** `AuthenticationError` if credentials are invalid
```

**Bad:**
```markdown
## authenticate()

To authenticate a user, call this function with their username
and password. This is useful when you need to verify who the
user is before giving them access to protected resources...
```

### 2. Be Comprehensive

Document everything, even if it seems obvious.

**Good:**
```markdown
## Configuration Options

- `port` (number): Server port. Default: 3000
- `host` (string): Server hostname. Default: 'localhost'
- `timeout` (number): Request timeout in ms. Default: 30000
- `debug` (boolean): Enable debug mode. Default: false
```

**Bad:**
```markdown
## Configuration Options

- `port`: The port number
- Other options are available
```

### 3. Structure According to the Product

Organize reference material to match the product's architecture.

**Good (API Reference):**
```markdown
# API Reference

## Authentication
- POST /auth/login
- POST /auth/logout
- POST /auth/refresh

## Users
- GET /users
- GET /users/:id
- POST /users
- PUT /users/:id
- DELETE /users/:id

## Posts
- GET /posts
- GET /posts/:id
- POST /posts
```

**Bad:**
```markdown
# API Reference

Endpoints in alphabetical order:
- DELETE /users/:id
- GET /posts
- GET /posts/:id
- GET /users
- POST /auth/login
```

### 4. Maintain Neutrality

Avoid opinions, recommendations, or guidance. Just state facts.

**Good:**
```markdown
## cache.invalidate(key)

Removes the entry with the specified key from the cache.

**Parameters:**
- `key` (string): Cache key to invalidate

**Returns:** (boolean) `true` if entry was found and removed, `false` otherwise
```

**Bad:**
```markdown
## cache.invalidate(key)

You should use this function when you want to remove stale data
from the cache. It's particularly useful in scenarios where...
```

### 5. Be Accurate and Precise

Every detail matters. Be exact.

**Good:**
```markdown
**Returns:** (number | null) User ID if found, null if not found

**Throws:**
- `ValidationError` if input is invalid
- `DatabaseError` if database connection fails
```

**Bad:**
```markdown
**Returns:** The user ID or nothing if not found

**Throws:** Various errors depending on what goes wrong
```

### 6. Use Consistent Patterns

Apply the same structure to all similar items.

**Good:**
```markdown
## add(a, b)
Returns the sum of two numbers.

## subtract(a, b)
Returns the difference of two numbers.

## multiply(a, b)
Returns the product of two numbers.
```

**Bad:**
```markdown
## add(a, b)
Returns the sum of two numbers.

Parameters: a and b are numbers

## subtract
Subtracts one number from another

## multiply(a, b)
a - first number
b - second number
Returns: a * b
```

## Structure of Reference Documentation

### 1. Overview (Brief)

State what is being documented.

**Example:**
```markdown
# Logger API Reference

The Logger module provides methods for structured application logging.
```

### 2. Organization

Group related items logically, following the product's structure.

**For APIs:**
- Group by resource or module
- List methods/endpoints under each

**For Configuration:**
- Group by section or category
- List all options systematically

**For CLI:**
- Group by command category
- List all commands and flags

### 3. Each Item (Consistent Structure)

For each documented item, provide:

**Functions/Methods:**
```markdown
## functionName(params)

Brief description of what it does.

**Parameters:**
- `param1` (type): Description
- `param2` (type, optional): Description. Default: value

**Returns:** (type) Description

**Throws:**
- `ErrorType1`: When condition
- `ErrorType2`: When condition

**Example:**
```javascript
const result = functionName(value1, value2);
```
```

**API Endpoints:**
```markdown
## POST /api/resource

Brief description of what this endpoint does.

**Headers:**
- `Authorization`: Bearer token (required)
- `Content-Type`: application/json

**Request Body:**
```json
{
  "field1": "string",
  "field2": "number"
}
```

**Response:** 200 OK
```json
{
  "id": "string",
  "created": "timestamp"
}
```

**Errors:**
- `400 Bad Request`: Invalid input
- `401 Unauthorized`: Missing or invalid token
- `409 Conflict`: Resource already exists
```

**Configuration Options:**
```markdown
## option_name

**Type:** string | number | boolean
**Default:** default_value
**Required:** yes/no

Description of what this option controls and its effect.

**Example:**
```yaml
option_name: example_value
```
```

**CLI Commands:**
```markdown
## command [options] [arguments]

Brief description of what the command does.

**Arguments:**
- `arg1`: Description

**Options:**
- `-f, --flag`: Description
- `-o, --option <value>`: Description

**Examples:**
```bash
command --flag arg1
command --option value arg1
```
```

## Writing Style

### Use Technical Language

Be precise and use correct terminology.

**Good:** "Returns a Promise that resolves to an Array of Objects"
**Bad:** "Gives you a list of things"

### Be Concise But Complete

Don't add fluff, but include all necessary information.

**Good:**
```markdown
**timeout** (number): Maximum time in milliseconds to wait
for response. Default: 5000
```

**Bad:**
```markdown
**timeout**: This is a really important setting that controls
how long the system will wait before giving up. You might want
to adjust this based on your network conditions...
```

### Use Present Tense

Describe what something does, not what it did or will do.

**Good:** "Returns the user object"
**Bad:** "Will return the user object"

### Be Literal

State exactly what happens.

**Good:** "Throws `ValidationError` if email format is invalid"
**Bad:** "Throws an error if something's wrong with the email"

### Show Types Clearly

Make data types explicit.

**Good:**
```markdown
**Parameters:**
- `id` (number | string): User identifier
- `options` (Object, optional): Configuration options
  - `includeDeleted` (boolean): Include deleted users. Default: false
```

**Bad:**
```markdown
**Parameters:**
- `id`: The user ID
- `options`: Some optional settings
```

## Best Practices

### ✓ DO

- **Document everything**: All functions, all parameters, all options
- **Be consistent**: Use same format for all similar items
- **Provide examples**: Show actual usage, but don't explain
- **Include edge cases**: Document behavior for unusual inputs
- **Specify defaults**: State default values clearly
- **List all errors**: Document all possible exceptions/errors
- **Use tables**: Great for parameter lists and option comparisons
- **Show type information**: Be explicit about data types
- **Update with code**: Keep reference in sync with implementation
- **Make it searchable**: Use clear, predictable naming

### ✗ DON'T

- **Don't explain concepts**: Link to explanations instead
- **Don't provide tutorials**: Link to tutorials instead
- **Don't give advice**: Stay neutral and objective
- **Don't be selective**: Document all features equally
- **Don't add opinions**: No "best practices" or recommendations
- **Don't skip "obvious" things**: Document comprehensively
- **Don't use marketing language**: Technical description only
- **Don't mix with how-to**: Keep pure reference separate

## Common Patterns

### Pattern 1: API Reference
```markdown
# API Reference

## Resource Name

Brief description of the resource.

### GET /api/resource
[detailed documentation]

### POST /api/resource
[detailed documentation]

### GET /api/resource/:id
[detailed documentation]
```

### Pattern 2: Function/Method Reference
```markdown
# Module Name

Brief description of the module.

## Methods

### method1(params)
[detailed documentation]

### method2(params)
[detailed documentation]
```

### Pattern 3: Configuration Reference
```markdown
# Configuration Reference

## Section Name

### option1
[detailed documentation]

### option2
[detailed documentation]
```

### Pattern 4: CLI Reference
```markdown
# Command Line Reference

## Commands

### command1 [options]
[detailed documentation]

### command2 [options]
[detailed documentation]
```

## Example: Good Reference Documentation

```markdown
# Cache API Reference

The Cache API provides methods for storing and retrieving data in memory.

## Methods

### set(key, value, options)

Stores a value in the cache with the specified key.

**Parameters:**
- `key` (string): Cache key. Must be non-empty.
- `value` (any): Value to store. Must be serializable to JSON.
- `options` (Object, optional): Storage options
  - `ttl` (number, optional): Time-to-live in seconds. Default: 3600
  - `tags` (Array<string>, optional): Tags for cache invalidation. Default: []

**Returns:** (boolean) `true` if stored successfully, `false` if key already exists

**Throws:**
- `InvalidKeyError`: If key is empty or not a string
- `SerializationError`: If value cannot be serialized to JSON

**Example:**
```javascript
cache.set('user:123', { name: 'John' }, { ttl: 7200 });
// Returns: true
```

---

### get(key)

Retrieves a value from the cache.

**Parameters:**
- `key` (string): Cache key

**Returns:** (any | null) Cached value if found and not expired, `null` otherwise

**Example:**
```javascript
const user = cache.get('user:123');
// Returns: { name: 'John' } or null
```

---

### delete(key)

Removes an entry from the cache.

**Parameters:**
- `key` (string): Cache key

**Returns:** (boolean) `true` if entry existed and was deleted, `false` otherwise

**Example:**
```javascript
cache.delete('user:123');
// Returns: true
```

---

### clear(tags)

Removes all entries from the cache, or entries with specified tags.

**Parameters:**
- `tags` (Array<string>, optional): Only clear entries with these tags. If omitted, clears all entries.

**Returns:** (number) Number of entries removed

**Example:**
```javascript
cache.clear(['users']);
// Returns: 42
```

---

### has(key)

Checks if a key exists in the cache and has not expired.

**Parameters:**
- `key` (string): Cache key

**Returns:** (boolean) `true` if key exists and is not expired, `false` otherwise

**Example:**
```javascript
if (cache.has('user:123')) {
  // Use cached value
}
```

---

## Configuration

### maxSize

**Type:** number
**Default:** 1000
**Required:** no

Maximum number of entries the cache can hold. When exceeded, oldest entries are evicted.

**Example:**
```javascript
const cache = new Cache({ maxSize: 5000 });
```

---

### defaultTTL

**Type:** number
**Default:** 3600
**Required:** no

Default time-to-live in seconds for cache entries when not specified in `set()`.

**Example:**
```javascript
const cache = new Cache({ defaultTTL: 7200 });
```

---

### onEvict

**Type:** function
**Default:** undefined
**Required:** no

Callback function invoked when an entry is evicted.

**Signature:** `(key: string, value: any) => void`

**Example:**
```javascript
const cache = new Cache({
  onEvict: (key, value) => {
    console.log(`Evicted ${key}`);
  }
});
```
```

## Checklist for Reference Documentation Quality

Before publishing, verify:

- [ ] Every public function/method/endpoint is documented
- [ ] All parameters are listed with types
- [ ] Return types are specified
- [ ] All possible errors/exceptions are documented
- [ ] Default values are stated
- [ ] Optional vs. required is clear
- [ ] Structure mirrors the product's architecture
- [ ] Consistent format used throughout
- [ ] Examples are provided (but not explained)
- [ ] No opinions or recommendations (pure facts)
- [ ] No instructional content (no "how to")
- [ ] No explanatory content (no "why")
- [ ] Terminology is accurate and precise
- [ ] Content is comprehensive (nothing missing)
- [ ] Easy to scan and search

## Common Mistakes

### Mistake: Mixing in Instructions
**Bad:** "To authenticate, call `auth()` with your credentials"
**Good:** "`auth(credentials)` validates credentials and returns a token"

### Mistake: Adding Opinions
**Bad:** "The recommended timeout is 30 seconds"
**Good:** "`timeout` (number): Request timeout in seconds. Default: 30"

### Mistake: Being Incomplete
**Bad:**
```markdown
## process(data)
Processes the data
```
**Good:**
```markdown
## process(data)
Validates and transforms input data according to configured rules.

**Parameters:**
- `data` (Object): Input data to process

**Returns:** (Object) Transformed data

**Throws:** `ValidationError` if data is invalid
```

### Mistake: Inconsistent Structure
**Bad:** Different functions documented with different formats
**Good:** Every function follows the exact same structure

### Mistake: Explaining Instead of Describing
**Bad:** "This function is useful when you need to..."
**Good:** "Returns the sum of two numbers"

### Mistake: Hiding Information in Prose
**Bad:** "The function takes a user ID (number) and returns their profile"
**Good:** Use proper parameter and return type documentation structure

---

**Remember**: Reference documentation is a technical description of the machinery. Be accurate, be complete, be neutral, be consistent. Your users need facts they can trust and find quickly.
