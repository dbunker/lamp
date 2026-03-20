%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing                          : 0.008s
%% Hypothesis Space Generation             : 0.149s
%% Conflict analysis                       : 0.366s
%%   - Positive Examples                   : 0.366s
%% Counterexample search                   : 0.002s
%%   - CDOEs                               : 0s
%%   - CDPIs                               : 0s
%% Hypothesis Search                       : 0.098s
%% Total                                   : 0.643s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d4(V1) :- d1(V1).
d5(V1) :- d0(V1).
d0(V1) :- d4(V1).

