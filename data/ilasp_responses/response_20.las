%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing                          : 0.007s
%% Hypothesis Space Generation             : 0.601s
%% Conflict analysis                       : 2.922s
%%   - Positive Examples                   : 2.922s
%% Counterexample search                   : 0.002s
%%   - CDOEs                               : 0s
%%   - CDPIs                               : 0.001s
%% Hypothesis Search                       : 0.756s
%% Total                                   : 4.457s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d2(V1) :- d9(V1).
d3(V1) :- d7(V1).
d7(V1) :- d8(V1); not d5(V1).
d5(V1) :- d2(V1); not d8(V1).

