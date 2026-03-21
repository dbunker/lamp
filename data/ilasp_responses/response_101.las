%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing                          : 0.008s
%% Hypothesis Space Generation             : 0.067s
%% Conflict analysis                       : 0.093s
%%   - Positive Examples                   : 0.093s
%% Counterexample search                   : 0.001s
%%   - CDOEs                               : 0s
%%   - CDPIs                               : 0s
%% Hypothesis Search                       : 0.035s
%% Total                                   : 0.211s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d0(V1) :- d5(V1).
d4(V1) :- d0(V1).
d0(V1) :- d2(V1); not d3(V1).

