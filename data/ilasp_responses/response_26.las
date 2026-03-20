%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing                          : 0.008s
%% Hypothesis Space Generation             : 1.382s
%% Conflict analysis                       : 43.576s
%%   - Positive Examples                   : 43.576s
%% Counterexample search                   : 0.002s
%%   - CDOEs                               : 0s
%%   - CDPIs                               : 0.002s
%% Hypothesis Search                       : 14.822s
%% Total                                   : 60.583s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d1(V1) :- d8(V1).
d2(V1) :- d7(V1).
d3(V1) :- d7(V1).
d8(V1) :- d7(V1).
d9(V1) :- d5(V1).
d3(V1) :- d4(V1).
d0(V1) :- d2(V1); not d7(V1).

