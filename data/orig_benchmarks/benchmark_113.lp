d4(o0).
d4(o1).
d1(o0).
d2(o1).
d0(o3).

d0(X) :- d2(X), d1(X), not d3(X).
d4(X) :- d2(X), d5(X), not d0(X).
d4(X) :- d1(X), not d3(X), d5(X).
d5(X) :- d2(X), d0(X), not d4(X).
d3(X) :- d4(X), d1(X), not d5(X).
d1(X) :- d0(X), d5(X), not d2(X).
d1(X) :- d0(X), d3(X), not d4(X).
d4(X) :- not d5(X), d0(X), d3(X).
d5(X) :- d2(X), d3(X), not d1(X).
d3(X) :- d2(X), not d0(X), d4(X).
d4(X) :- d2(X), d0(X), not d3(X).