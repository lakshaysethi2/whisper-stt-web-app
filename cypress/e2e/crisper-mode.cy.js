describe('CrisperWhisper: transcription style selector', () => {
  it('shows the mode selector for crisperwhisper models only', () => {
    cy.visit('/');

    // Mode group hidden until a crisperwhisper model is picked
    cy.get('#mode-group').should('not.be.visible');

    // faster-whisper models keep it hidden
    cy.get('#model-select').select('base');
    cy.get('#mode-group').should('not.be.visible');

    // crisperwhisper model shows it, defaulting to verbatim
    cy.get('#model-select').select('crisperwhisper-small');
    cy.get('#mode-group').should('be.visible');
    cy.get('#mode-select').should('have.value', 'verbatim');

    // switching modes works
    cy.get('#mode-select').select('intended');
    cy.get('#mode-select').should('have.value', 'intended');

    // clearing the model hides the mode group again
    cy.get('#model-select').select('');
    cy.get('#mode-group').should('not.be.visible');
  });
});
