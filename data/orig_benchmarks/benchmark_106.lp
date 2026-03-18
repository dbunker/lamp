d5(o3).
d0(o2).
d1(o3).
d4(o4).
d0(o4).
d3(o0).

d2(X) :- d1(X), not d3(X).
d1(X) :- d0(X), not d2(X).
d1(X) :- d0(X), not d2(X).
d0(X) :- d4(X), not d5(X).
d0(X) :- d1(X), not d5(X).
d1(X) :- d0(X), not d2(X).
d3(X) :- d5(X), not d4(X).
d3(X) :- d0(X), not d2(X).
d2(X) :- not d3(X), d5(X).
d1(X) :- d4(X), not d5(X).
d3(X) :- d2(X), not d5(X).