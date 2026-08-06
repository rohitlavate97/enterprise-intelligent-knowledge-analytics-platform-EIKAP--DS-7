def test_ticket_classification_and_urgency(ticket_classifier, sample_tickets):
    for ticket in sample_tickets:
        result = ticket_classifier.classify_and_urgency(ticket["text"])
        assert "category" in result
        assert "urgency" in result
        
        if ticket["label"] == "access_issue":
            assert result["category"] == "access_issue"
            assert result["urgency"] == "high"
        elif ticket["label"] == "billing":
            assert result["category"] == "billing"
            assert result["urgency"] == "normal"
