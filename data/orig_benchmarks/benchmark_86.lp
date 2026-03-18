d4(o0).
d4(o1).
d1(o5).
d0(o5).
d2(o1).

d4(X) :- not d3(X), d5(X).
d0(X) :- d1(X), not d3(X).
d3(X) :- d0(X), not d5(X).
d0(X) :- d5(X), not d4(X).
d0(X) :- d1(X), not d4(X).
d3(X) :- d4(X), not d1(X).
d5(X) :- d2(X), not d0(X).
d3(X) :- d4(X), not d0(X).
d1(X) :- d0(X), not d4(X).
d5(X) :- d0(X), not d2(X).
d2(X) :- not d1(X), d0(X).