# AmazonHelp AI Support Agent

AI customer-support agent built for the **Hiver SDE Intern take-home assignment**.

##  Live Demo

**Streamlit app:**  
https://hiver-amazonapp-agent-mbpjvg7jbtk2vepeq3jhs4.streamlit.app/

The interactive deployment uses a compact **500-row temporal-safe historical subset** for demonstration.

The reported evaluation metrics were produced using the full retrieval corpus.

---

## Pipeline

```text
Customer Message
       ↓
Intent Classification
       ↓
Dense Historical Retrieval
       ↓
Evidence Assessment
       ↓
Grounded Response Generation
       ↓
Human Escalation Decision
