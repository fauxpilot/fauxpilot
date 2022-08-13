// The module 'vscode' contains the VS Code extensibility API
// Import the module and reference it with the alias vscode in your code below
import * as vscode from 'vscode';
import axios from 'axios';

const handleGetCompletions = async (
textContext: string
): Promise<Array<any>> => {

const completions = await axios
    .post(
    `http://127.0.0.1:5000/v1/engines/codegen/completions`,
    {
        prompt: textContext,
        max_tokens: 16,
        temperature: 0.1, 
        stop: ["\n\n"]
    },
    {
        headers: {
        'Authorization': 'Bearer dummy',
        'Content-Type': 'application/json'
        }
    }
    )
    .then((res) => res.data.choices)
    .catch(async (err: Error) => {
    vscode.window.showErrorMessage(err.toString());
    });

return completions;
};


export function activate(_: vscode.ExtensionContext) {

    const provider: vscode.CompletionItemProvider = {
        // @ts-ignore
        provideInlineCompletionItems: async (document, position, context, token) => {

            // The text on the line
            const textBeforeCursor: string = document.getText(
                new vscode.Range(position.with(undefined, 0), position)
            );

            if (textBeforeCursor.trim().length < 3) {
              // Do not predict for lines with less than 3 characters
              return []
            }

            let completions = await handleGetCompletions(
                textBeforeCursor
              ).catch((err) =>
                vscode.window.showErrorMessage(err.toString())
              ) as Array<any>;

            return completions.map((completion_result: any) => ({
                insertText: completion_result.text,
                range: new vscode.Range(position, position.translate(undefined, completion_result.text.length))
                }));
        },
    };

    // @ts-ignore
    vscode.languages.registerInlineCompletionItemProvider({pattern: '**'}, provider);
}

// this method is called when your extension is deactivated
export function deactivate() {}
