d5(o0).
d0(o0).
d3(o6).
d1(o9).
d3(o3).
d0(o9).
d4(o1).
d1(o3).
d0(o3).

d5(X) :- d0(X), not d4(X).
d5(X) :- d1(X), not d4(X).
d5(X) :- d0(X), not d3(X).
d2(X) :- d0(X), not d5(X).
d4(X) :- not d2(X), d3(X).
d4(X) :- not d2(X), d3(X).
d1(X) :- d0(X), not d4(X).
d0(X) :- not d3(X), d4(X).
d4(X) :- not d2(X), d5(X).
d3(X) :- not d2(X), d1(X).
d5(X) :- not d2(X), d1(X).