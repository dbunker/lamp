%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing                          : 0.007s
%% Hypothesis Space Generation             : 0.064s
%% Conflict analysis                       : 0.081s
%%   - Positive Examples                   : 0.081s
%% Counterexample search                   : 0.002s
%%   - CDOEs                               : 0s
%%   - CDPIs                               : 0s
%% Hypothesis Search                       : 0.023s
%% Total                                   : 0.183s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d5(V1) :- d3(V1).
d2(V1) :- d5(V1).

