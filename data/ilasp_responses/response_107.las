%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing                          : 0.008s
%% Hypothesis Space Generation             : 0.188s
%% Conflict analysis                       : 0.946s
%%   - Positive Examples                   : 0.946s
%% Counterexample search                   : 0.002s
%%   - CDOEs                               : 0s
%%   - CDPIs                               : 0.002s
%% Hypothesis Search                       : 0.371s
%% Total                                   : 1.55s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d2(V1) :- d0(V1).
d0(V1) :- d3(V1).
d0(V1) :- d4(V1).
d5(V1) :- d0(V1).
d4(V1) :- d2(V1); not d3(V1).

