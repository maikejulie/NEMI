# Writing Tutorials

## What is a Tutorial?

A tutorial is **learning-oriented** documentation that guides a beginner through a complete, practical activity to acquire basic skills and confidence.

**Key characteristics:**
- Learning by doing
- Provides a guided lesson
- Takes the user by the hand
- Focuses on skill acquisition
- Guarantees success

**User context:**
- New to the tool/technology
- Learning mode (study, not work)
- Needs to build confidence
- Wants to see what's possible

**User question:** "Can you teach me to...?"

## Purpose and Goals

### Primary Purpose
Enable a beginner to acquire basic skills and confidence through a successful, hands-on experience.

### Success Criteria
- User completes the tutorial successfully
- User gains confidence in using the tool
- User understands what's possible
- User is motivated to continue learning
- User has working knowledge to build on

### What Tutorials Are NOT
- Not reference material (that's Reference documentation)
- Not problem-solving guides (that's How-to Guides)
- Not comprehensive teaching (focus on basics)
- Not explanatory (that's Explanation documentation)
- Not for experienced users (they need How-to Guides)

## Core Principles

### 1. Learning by Doing

Tutorials focus on **concrete actions** that produce **visible results**.

**Good:**
```markdown
Create a new file called `hello.py`:
```python
print("Hello, world!")
```

Run the file:
```bash
python hello.py
```

You should see: `Hello, world!`
```

**Bad:**
```markdown
Python is a high-level programming language with dynamic typing.
The print function outputs text to stdout. Let's explore different
ways you might want to use print...
```

### 2. Provide Immediate Success

Start with the simplest possible successful outcome, as quickly as possible.

**Good:** "In 5 minutes, you'll have a working web server"
**Bad:** "First, let's understand HTTP protocols and server architecture..."

### 3. Ensure Reliability

Every step must work exactly as described. No surprises, no failures.

**Requirements:**
- Test the tutorial thoroughly
- Specify exact versions if needed
- Provide troubleshooting for common issues
- Ensure prerequisites are clear
- Make no assumptions about user knowledge

### 4. Guide, Don't Explore

Make all decisions for the learner. No options, no alternatives, no "you could also..."

**Good:**
```markdown
Set the timeout to 30 seconds:
```python
timeout = 30
```
```

**Bad:**
```markdown
Set the timeout to whatever value works for your use case.
Common values are 30, 60, or 300 seconds depending on
whether you're dealing with fast local operations or
slow external APIs...
```

### 5. Minimize Explanation

Focus on **what** to do, not **why** it works. Link to explanations instead.

**Good:**
```markdown
Add this middleware to enable authentication:
```python
app.use(authenticate)
```

[Learn more about authentication →](link-to-explanation)
```

**Bad:**
```markdown
Add this middleware to enable authentication. Middleware
functions are executed sequentially in the order they're
registered. The authenticate middleware checks the session
cookie, validates it against the database, and attaches
the user object to the request...
```

### 6. Repeat and Reinforce

Let learners practice the same skills in slightly different contexts.

**Example:**
```markdown
1. Create a user: `python manage.py createuser`
2. Create a group: `python manage.py creategroup`
3. Create a project: `python manage.py createproject`
```

(All three commands follow the same pattern)

## Structure of a Tutorial

### 1. Introduction

**What to include:**
- What the user will learn
- What the user will build/achieve
- How long it will take
- Prerequisites (keep minimal)

**Example:**
```markdown
# Your First Web API

In this tutorial, you'll build a simple REST API that manages
a todo list. You'll learn to:
- Set up a new project
- Create API endpoints
- Handle data with a database
- Test your API

**Time**: 30 minutes
**Prerequisites**: Python 3.8+ installed
```

### 2. Setup

**What to include:**
- Installation instructions
- Initial project setup
- Verification that setup worked

**Example:**
```markdown
## Setup

Install the framework:
```bash
pip install webframework
```

Create a new project:
```bash
webframework new myproject
cd myproject
```

Verify it works:
```bash
webframework run
```

You should see: `Server running on http://localhost:8000`
```

### 3. Main Content (Step-by-Step)

**Structure each step as:**
1. Clear instruction
2. Exact code or command
3. Expected result
4. Brief confirmation

**Example:**
```markdown
## Step 1: Create Your First Endpoint

Open `app.py` and add this code:
```python
@app.route('/hello')
def hello():
    return {'message': 'Hello, World!'}
```

Start the server:
```bash
webframework run
```

Visit http://localhost:8000/hello in your browser.
You should see: `{"message": "Hello, World!"}`

✓ You've created your first API endpoint!
```

### 4. Conclusion

**What to include:**
- Recap what was learned
- What the user has achieved
- What they can do next
- Links to related how-to guides or explanations

**Example:**
```markdown
## What You've Learned

Congratulations! You've built a working REST API. You now know how to:
- ✓ Set up a new project
- ✓ Create API endpoints
- ✓ Store data in a database
- ✓ Test your API

## Next Steps

Now that you have the basics, you can:
- [Add authentication to your API →](link-to-how-to)
- [Deploy your API to production →](link-to-how-to)
- [Understand how routing works →](link-to-explanation)
```

## Writing Style

### Use Imperative Mood
Tell the user what to do directly.

**Good:** "Create a file," "Run the command," "Add this code"
**Bad:** "You should create," "We can run," "Let's add"

### Be Concrete and Specific
Provide exact commands, exact file names, exact content.

**Good:** "Create `config.yaml` with these exact contents:"
**Bad:** "Create a configuration file"

### Use Short Sentences
Keep instructions clear and scannable.

**Good:** "Save the file. Run the server. Open your browser."
**Bad:** "After saving the file, you'll want to run the server, then navigate to your browser."

### Provide Feedback
Tell users what they should see after each step.

**Good:**
```markdown
Run `npm start`
You should see: `Compiled successfully!`
```

**Bad:**
```markdown
Run `npm start`
```

### Use Encouraging Language
Build confidence without being condescending.

**Good:** "Great! Your server is running."
**Bad:** "Wow! You're amazing! You did it!"

## Best Practices

### ✓ DO

- **Focus on one clear goal**: "Build a todo list app"
- **Start simple**: Most basic successful outcome first
- **Build progressively**: Each step adds one new concept
- **Show, don't tell**: Code examples, not prose
- **Verify frequently**: User confirms success after each step
- **Handle errors**: Provide troubleshooting for common issues
- **Test thoroughly**: Every single step, multiple times
- **Use realistic examples**: Meaningful, not abstract
- **Provide working code**: No "...", no placeholders
- **Make it meaningful**: Build something useful

### ✗ DON'T

- **Don't explain everything**: Link to explanations instead
- **Don't offer choices**: Make decisions for the learner
- **Don't assume knowledge**: Specify all prerequisites
- **Don't skip steps**: Every action must be explicit
- **Don't leave gaps**: No "now configure the settings..."
- **Don't use placeholders**: No "YOUR_API_KEY_HERE"
- **Don't be abstract**: Use real, concrete examples
- **Don't optimize early**: Simple working code first
- **Don't distract**: Stay focused on the learning path
- **Don't rush**: Take time to build confidence

## Common Patterns

### Pattern 1: Installation Tutorial
```markdown
# Getting Started with X

Install X, create your first project, verify it works.

1. Install
2. Create new project
3. Run "hello world"
4. Verify output
```

### Pattern 2: Build a Simple Project
```markdown
# Build Your First Y

Create a complete, minimal working Y application.

1. Set up project
2. Add core functionality
3. Add one feature at a time
4. Test the complete application
```

### Pattern 3: Hands-On Introduction
```markdown
# Introduction to Z

Learn Z concepts through practical exercises.

1. Exercise 1: Basic concept
2. Exercise 2: Build on concept
3. Exercise 3: Combine concepts
4. Review what you've learned
```

## Example: Good Tutorial

```markdown
# Build Your First Blog

Create a simple blog with posts and comments in 20 minutes.

## Prerequisites
- Python 3.8+
- Basic command line knowledge

## Setup

Install BlogFramework:
```bash
pip install blogframework
```

Create a new blog:
```bash
blog new myblog
cd myblog
```

Start the development server:
```bash
blog serve
```

Visit http://localhost:8000. You should see the default homepage.

## Create Your First Post

Create a new post:
```bash
blog post create
```

When prompted:
- Title: `My First Post`
- Author: `Your Name`

Edit `posts/my-first-post.md` and add:
```markdown
This is my first blog post!
```

Refresh http://localhost:8000. You should see your post on the homepage.

Click the post title to see the full post.

## Add a Comment

Open http://localhost:8000/posts/my-first-post

Scroll to the comments section and add:
- Name: `Test User`
- Comment: `Great post!`

Click "Submit". Your comment should appear below the post.

## Customize the Theme

Edit `config.yaml`:
```yaml
theme:
  color: blue
  title: My Awesome Blog
```

Refresh the page. The site should now have a blue theme with your custom title.

## What You've Learned

You've built a working blog! You can now:
- ✓ Create and publish posts
- ✓ Allow comments
- ✓ Customize the theme

## Next Steps
- [Add authentication →](link)
- [Deploy your blog →](link)
- [Understand the architecture →](link)
```

## Checklist for Tutorial Quality

Before publishing, verify:

- [ ] Prerequisites are clearly stated and minimal
- [ ] Every step works exactly as described
- [ ] User gets successful result early (within 5 minutes)
- [ ] Each step has expected output specified
- [ ] No unexplained jargon or terms
- [ ] No choices or alternatives presented
- [ ] Explanations are minimal and link to separate docs
- [ ] Progression is logical and builds skills gradually
- [ ] The completed result is useful and meaningful
- [ ] Next steps are provided at the end
- [ ] Tested on a clean environment multiple times

## Common Mistakes

### Mistake: Over-Explaining
**Bad:** "We use JSON because it's a lightweight data format that..."
**Good:** "Return the data as JSON:" (link to explanation)

### Mistake: Offering Options
**Bad:** "You can use SQLite, PostgreSQL, or MySQL depending on..."
**Good:** "Use SQLite for this tutorial:"

### Mistake: Assuming Knowledge
**Bad:** "Configure your environment variables"
**Good:** "Create a `.env` file with these contents:"

### Mistake: Skipping Verification
**Bad:** "Run the server."
**Good:** "Run the server. You should see `Server started on port 8000`"

### Mistake: Being Too Abstract
**Bad:** "Create a model for your domain objects"
**Good:** "Create a Post model with title and content fields"

---

**Remember**: A tutorial's job is to give someone their first successful experience with your tool. Everything else is secondary. Keep it simple, keep it reliable, and keep them succeeding.
