describe('Model guard: record button disabled until model selected', () => {
  it('disables record button initially and enables on model selection', () => {
    cy.visit('/');

    // Record button should be disabled initially
    cy.get('#record-btn').should('be.disabled');

    // Hint should show model selection message
    cy.get('#record-hint').should('contain.text', 'Select a model');

    // Select a model
    cy.get('#model-select').select('base');

    // Record button should now be enabled
    cy.get('#record-btn').should('not.be.disabled');

    // Hint should update
    cy.get('#record-hint').should('contain.text', 'Tap to start recording');

    // Switch model back to empty
    cy.get('#model-select').select('');

    // Record button should be disabled again
    cy.get('#record-btn').should('be.disabled');
    cy.get('#record-hint').should('contain.text', 'Select a model');
  });
});
